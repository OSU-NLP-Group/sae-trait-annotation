import json
import logging
import os
import random
import collections
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import numpy as np
import saev
import saev.activations
import saev.nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams
from torchvision.transforms import v2
from together import Together
import base64
from io import BytesIO
import requests

logger = logging.getLogger("visuals")


def load_imagefolder_dataset(args, path, batch_size=32, debug=False):
    dataset = datasets.ImageFolder(root=path)  # No transform!
    
    if debug:
        subset_indices = list(range(args.n_debug_ex))
        class_names = dataset.classes
        dataset = Subset(dataset, subset_indices)
        dataset.classes = class_names

    def collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataset, loader

# Extract embeddings for a dataset
def extract_embeddings(args, dataloader, recorded_vit, sae, img_transform, device = "cuda"):
    all_embeddings = []
    all_labels = []
    transforms.ToTensor()

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            
            images_t = img_transform(images)

            images_t = torch.stack(images_t).to(device)

            _, vit_acts = recorded_vit(images_t)

            # Select the only layer in the batch and ignore the CLS token.
            vit_acts = vit_acts[:, 0, 1:, :]

            _, f_x, _ = sae(vit_acts)

            all_embeddings.extend(torch.unbind(f_x, dim=0))
            all_labels.extend(labels.tolist())

    print('flag 01')
    print('len(all_embeddings) = {}'.format(len(all_embeddings)))
    print('len(all_labels) = {}'.format(len(all_labels)))

    return all_embeddings, all_labels

def patch_index_to_image_coordinates(patch_index, patch_size=14, num_patches_per_row=16):
    """
    Maps a patch index to (x, y) pixel coordinates of the top-left corner of the patch in the image.

    Args:
        patch_index (int): Index of the patch (0 to 195 for ViT-B).
        patch_size (int): Size of each patch in pixels (default 16).
        num_patches_per_row (int): Number of patches in one row (default 14 for 224x224 images).

    Returns:
        (x, y): Top-left pixel coordinates in the original image.
    """
    row = patch_index // num_patches_per_row
    col = patch_index % num_patches_per_row
    x = col * patch_size
    y = row * patch_size
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/fs/ess/PAS2136/user/bioscan-5m/split_50_1000')
    parser.add_argument('--out-dir', type=str, required=True, help='output directory for activations')
    parser.add_argument('--sae-ckpt-path', type=str, default='/home/user/research_nfs/visual_barcoding/sae_ckpt/checkpoints/9pzqonq0/sae.pt')
    parser.add_argument('--debug', action='store_true', help='use data subset for debugging')
    parser.add_argument('--n-patches', type=int, default=256, help='No. of patches for ViT in SAE')
    parser.add_argument('--layer-id', type=int, default=10, help='layer ID for ViT in SAE')
    parser.add_argument('--thresh', type=float, default=0.9, help='threshold for SAE features')
    parser.add_argument('--trait-thresh', type=float, default=1e-4, help='threshold for trait to be considered as prominent at species/genus level')
    parser.add_argument('--patch-size', type=int, default=14, help='size of the patch')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--serve-choice', choices=['qwen_hf', 'qwen_vllm', 'llama3.2', 'qwen_72b'], default='qwen_hf', help='choice of model serving')
    parser.add_argument('--omit-label-vlm', action='store_true', help='omit the species label in image_wo_latent input to VLM', default=True)
    parser.add_argument('--api-url', type=str, default="http://0.0.0.0:8753/v1/chat/completions", help='API URL for Qwen 72B model')
    parser.add_argument('--n-debug-ex', type=int, default=1000, help='no of examples in debug mode')
    parser.add_argument('--n-img-input', type=int, default=3, help='no of images to input to VLM')

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load selected model and processor
    img_transform = saev.activations.make_img_transform("dinov2", args.sae_ckpt_path)

    print('img_transform = {}'.format(img_transform))

    vit = saev.activations.make_vit("dinov2", 'dinov2_vitb14')

    # layer 10 of 12 layers
    recorded_vit = saev.activations.RecordedVisionTransformer(vit, args.n_patches, True, [args.layer_id]).to(device)

    sae = saev.nn.load(args.sae_ckpt_path)
    print('sae = {}'.format(next(sae.parameters()).device))
    print('recorded_vit = {}'.format(next(recorded_vit.parameters()).device))

    train_dir = os.path.join(args.data_dir, 'train')

    # Load datasets
    train_dataset, train_loader = load_imagefolder_dataset(args, train_dir, debug=args.debug)
    
    print("Extracting training embeddings...")
    train_embeddings, train_labels = extract_embeddings(args, train_loader, recorded_vit, sae, img_transform, device)
    
    print('finished extracting training embeddings')
    
    # iterate over all possible genuses
    if hasattr(train_dataset, 'dataset'):
        class_to_idx = train_dataset.dataset.class_to_idx
    else:
        class_to_idx = train_dataset.class_to_idx

    id_to_class_map = {v:k for k,v in class_to_idx.items()}

    train_class_labels = [id_to_class_map[label] for label in train_labels]
    print("len(train_class_labels) = {}".format(len(train_class_labels)))

    train_genus_labels = [label.split('_')[0] for label in train_class_labels]
    train_species_labels = [' '.join(label.split('_')) for label in train_class_labels]

    # get set of unique genus labels
    unique_genus_labels = set(train_genus_labels)
    unique_species_labels = set(train_species_labels)

    print("Unique genus labels: {}".format(unique_genus_labels))
    print("Unique species labels: {}".format(unique_species_labels))
    print("Total number of unique genus labels: {}".format(len(unique_genus_labels)))
    print("Total number of unique species labels: {}".format(len(unique_species_labels)))
    
    # create a mapping from genus to empty counter
    genus_to_counter = {genus: collections.Counter() for genus in unique_genus_labels}
    species_to_counter = {species: collections.Counter() for species in unique_species_labels}
    
    print("Computing latents for each genus and species...")

    # find the corresponding images and compute their sae latents
    latent_to_patch_map = {}

    # find latents frequently expressed at genus level: store in a list
    for i in tqdm(range(len(train_embeddings))):
        # get the genus label for the current embedding
        genus_label = train_genus_labels[i]

        # get the species label for the current embedding
        species_label = train_species_labels[i]

        # get the embedding for the current image
        latent = train_embeddings[i].cpu() # [n_patch, sae_dim]

        # find the latents frequently expressed in this image (at any patch)
        latent_binary = (latent > args.thresh).numpy().astype(int)
        
        # find non zero indices of latents
        latent_binary_nonzero_idx = np.nonzero(latent_binary)
        non_zero_latents = latent_binary_nonzero_idx[1].tolist()
        
        latent_to_patch_map[i] = defaultdict(list)
        for j in range(len(latent_binary_nonzero_idx[0])):
            latent_to_patch_map[i][int(latent_binary_nonzero_idx[1][j])].append(int(latent_binary_nonzero_idx[0][j]))

        # update the counter for the current genus and species
        genus_to_counter[genus_label].update(non_zero_latents)
        species_to_counter[species_label].update(non_zero_latents)

    print('flag 3')

    # print('genus_to_counter = {}'.format(genus_to_counter))
    # print('species_to_counter = {}'.format(species_to_counter))

    os.makedirs(args.out_dir, exist_ok=True)

    # find latents frequently expressed at species but not genus level
    # find the latents that are frequently expressed at species level but not at genus level
    
    species_latents_prominent = defaultdict(list)
    species_freq_list = []
    genus_freq_list = []
    species_ratio_list = []
    genus_ratio_list = []

    for ex_id, train_label in enumerate(tqdm(train_class_labels)):
        # get the genus label for the current embedding
        genus_label = train_label.split('_')[0]

        # get the species label for the current embedding
        species_label = ' '.join(train_label.split('_'))

        genus_c = genus_to_counter[genus_label]
        species_c = species_to_counter[species_label]

        for latent_idx in latent_to_patch_map[ex_id]:
            species_freq = species_c[latent_idx]
            genus_freq = genus_c[latent_idx]
            species_ratio = species_freq*1.0/sum(species_c.values())
            genus_ratio = genus_freq*1.0/sum(genus_c.values())

            if species_ratio > genus_ratio:
                # print('species is greater, genus_freq = {}, species_freq = {}'.format(genus_freq, species_freq))
                # print('sum(species_c) = {}'.format(sum(species_c.values())))
                # print('sum(genus_c) = {}'.format(sum(genus_c.values())))
                species_freq_list.append(species_freq)
                genus_freq_list.append(genus_freq)

                species_ratio_list.append(species_ratio)
                genus_ratio_list.append(genus_ratio)

                # print('species_ratio = {}, genus_ratio = {}'.format(species_ratio, genus_ratio))

            # if the trait is prominent at species and genus level
            if species_ratio > args.trait_thresh and genus_ratio > args.trait_thresh and species_ratio > genus_ratio:
                # find the latents that are frequently expressed at species level but not at genus level
                
                # diff = species_freq - genus_freq

                # check if diff is significant
                # if len(diff) > 0:

                # store the latents that are frequently expressed at species level but not at genus level
                
                # store image index and patch index
                species_latents_prominent[species_label].append({'latent_idx': latent_idx, 'ex_id':ex_id})

    # print('species_latents_prominent = {}'.format(species_latents_prominent))

    # plot histogram of species_freq_list and genus_freq_list
    
    os.makedirs(os.path.join(args.out_dir, 'species_latents_prominent'), exist_ok=True)
    
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    if args.serve_choice == 'qwen_hf':
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        )
        model.to("cuda:0")
        
    elif args.serve_choice == 'qwen_vllm':
        model = LLM(model=model_name_or_path)

    elif args.serve_choice == 'llama3.2':
        client = Together()

    elif args.serve_choice == 'qwen_72b':
        url = args.api_url
        headers = {
            "Content-Type": "application/json"
        }

    else:
        raise ValueError("Invalid serve_choice. Choose from ['qwen_hf', 'qwen_vllm', 'llama3.2', 'qwen_72b']")

    f1 = open(os.path.join(args.out_dir, 'species_latents_prominent', 'latent_response.jsonl'), 'w')

    # verbalize patches to create a dataset
    for species_label in tqdm(species_latents_prominent):
        # get the list of latents for this species
        latents = species_latents_prominent[species_label]

        # iterate over the latents
        for latent in latents:
            # get the latent index
            latent_idx = latent['latent_idx']

            # get the image index
            ex_id = latent['ex_id']

            image = train_dataset[ex_id][0]
            # image_t = img_transform(image)

            transform = v2.Compose([
                # TODO: I bet this should be 256, 256, which is causing localization issues in non-square images.
                v2.Resize(size=256),
                v2.CenterCrop(size=(224, 224)),
                v2.ToPILImage(),
            ])
            image_t2 = transform(image)

            print('type(image_t2) = {}'.format(type(image_t2)))
            # print('image = {}'.format(image.size))
            
            # print('type(image_t) = {}'.format(type(image_t)))
            # print('image_t = {}'.format(image_t.size()))
            
            # save the patch
            # patch = einops.rearrange(patch, 'c h w -> h w c')
            # patch = Image.fromarray((patch.numpy() * 255).astype(np.uint8))
            os.makedirs(os.path.join(args.out_dir, 'species_latents_prominent', species_label.replace(' ', '_')), exist_ok=True)
            # patch.save(os.path.join(args.out_dir, 'species_latents_prominent', species_label, f'latent_{latent_idx}_ex_{ex_id}.png'))

            print('Image {}, Label: {}, key morphological trait: {}'.format(ex_id, species_label, latent_idx))

            # create an image which shows the original image, patch highlighted and the label

            output_path = os.path.join(args.out_dir, 'species_latents_prominent', species_label.replace(' ', '_'), f'latent_{latent_idx}_ex_{ex_id}_annotated.png')

            image_1 = image_t2.copy()
            image_1.save(output_path.replace('.png', '_image_1.png'))

            image_1 = image_1.resize((image_1.width * 2, image_1.height * 2), resample=Image.BICUBIC)

            if args.n_img_input == 3:
                # get the other 2 images as well
                all_ex = [latents[i]['ex_id'] for i in range(len(latents)) if latents[i]['ex_id'] != ex_id]
                chosen_ex = random.sample(all_ex, 2)

                image_2 = train_dataset[chosen_ex[0]][0]
                image_3 = train_dataset[chosen_ex[1]][0]

                image_2 = transform(image_2)
                image_3 = transform(image_3)

                image_2 = image_2.resize((image_2.width * 2, image_2.height * 2), resample=Image.BICUBIC)
                image_3 = image_3.resize((image_3.width * 2, image_3.height * 2), resample=Image.BICUBIC)

                image_2.save(output_path.replace('.png', '_image_2.png'))
                image_3.save(output_path.replace('.png', '_image_3.png'))

            # verbalize the patch
            # user_prompt = "Describe the red-highlighted patch in the context of the whole image."
            # user_prompt = "You are given the image of an insert. Briefly describe the body part of the insect which is highlighted in the red box." \

                user_prompt = """You are given three images of insects. Your task is to visually analyze them and extract observable morphological traits.

	1.	Identify the visible body parts of the insect (e.g., head, thorax, abdomen, legs, wings, antennae), *common in all three images*.
	2.	For each part, identify its morphological features — such as shape, size, color, texture, or distinct markings.
	3.	After analyzing all three images individually, list the morphological traits that are *common across all three insects*. *Only output traits that are visibly consistent across all images*.

*IMPORTANT*: Do not infer or assume information that is not directly observable. Avoid adding external knowledge."""
            
            else:
                user_prompt = """You are given an image of an insect specimen. Your task is to visually examine the insect and describe its observable morphological traits.

	1.	Identify the visible body parts of the insect (e.g., head, thorax, abdomen, legs, wings, antennae).
	2.	For each part, briefly describe the observable morphological features—such as shape, size, color, texture, or distinct markings—*based solely on what is visible in the image*.

*IMPORTANT*:
1. Do not infer or assume information that is not directly observable. Avoid adding external knowledge.
2. Keep your response concise and under 200 tokens."""
            
            # print('user_prompt = {}'.format(user_prompt))

            messages = []
            if args.n_img_input == 3:
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "text", "text": user_prompt}]})
            else:
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]})

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            if args.serve_choice == 'qwen_hf':
                
                # print('prompt = ', prompt)
                if args.n_img_input == 3:
                    inputs = processor(text=prompt, images=[image_1, image_2, image_3], padding=True, return_tensors="pt").to("cuda:0")
                else:
                    inputs = processor(text=prompt, images=[image_1], padding=True, return_tensors="pt").to("cuda:0")
                
                print('n_img_input = {}'.format(args.n_img_input))
                print('len(prompt) = {}'.format(len(prompt)))
                print('image_1 = {}'.format(image_1.size))
                if args.n_img_input == 3:
                    print('image_2 = {}'.format(image_2.size))
                    print('image_3 = {}'.format(image_3.size))

                num_tokens = inputs['input_ids'].shape[-1]
                print(f"Number of input tokens (including image): {num_tokens}")

                generation_args = { 
                    "max_new_tokens": 196, 
                    "temperature": 0.01, 
                    "do_sample": True, 
                }

                # print(generation_args)

                generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

                # remove input tokens 
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            elif args.serve_choice == 'qwen_vllm':
                # Use the VLLM model to generate a response
                sampling_params = SamplingParams(temperature=0.01, max_tokens=196)
                
                if args.n_img_input == 3:
                    response = model.generate({
                        'prompt': prompt,
                        "multi_modal_data": {"image": [image_1, image_2, image_3]},},
                        sampling_params=sampling_params
                    )
                else:
                    response = model.generate({
                        'prompt': prompt,
                        "multi_modal_data": {"image": image_1},},
                        sampling_params=sampling_params
                    )
                try:
                    response = response[0].outputs[0].text
                except Exception:
                    print(response)
            elif args.serve_choice == 'llama3.2':
                buffered = BytesIO()
                buffered_1, buffered_2, buffered_3 = BytesIO(), BytesIO(), BytesIO()
                image_1.save(buffered_1, format="PNG")
                image_1_base64 = base64.b64encode(buffered_1.getvalue()).decode("utf-8")

                if args.n_img_input == 3:
                    image_2.save(buffered_2, format="PNG")
                    image_3.save(buffered_3, format="PNG")
                
                    image_2_base64 = base64.b64encode(buffered_2.getvalue()).decode("utf-8")
                    image_3_base64 = base64.b64encode(buffered_3.getvalue()).decode("utf-8")

                    response = client.chat.completions.create(
                        model="meta-llama/Llama-Vision-Free",
                        temperature=0.01,
                        messages=[{"role": "user", "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_1_base64}"}},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_2_base64}"}},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_3_base64}"}},
                                        {"type": "text", "text": user_prompt}
                                    ],
                                    }],
                    )
                else:
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-Vision-Free",
                        temperature=0.01,
                        messages=[{"role": "user", "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_1_base64}"}},
                                        {"type": "text", "text": user_prompt}
                                    ],
                                    }],
                    )
                response = response.choices[0].message.content
            elif args.serve_choice == 'qwen_72b':
                # Encode the image to base64
                buffered_1, buffered_2, buffered_3 = BytesIO(), BytesIO(), BytesIO()
                image_1.save(buffered_1, format="PNG")
                image_1_base64 = base64.b64encode(buffered_1.getvalue()).decode("utf-8")

                if args.n_img_input == 3:
                    image_2.save(buffered_2, format="PNG")
                    image_3.save(buffered_3, format="PNG")
                
                    image_2_base64 = base64.b64encode(buffered_2.getvalue()).decode("utf-8")
                    image_3_base64 = base64.b64encode(buffered_3.getvalue()).decode("utf-8")

                    data = {
                        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_1_base64}"}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_2_base64}"}},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_3_base64}"}},
                                    {"type": "text", "text": user_prompt}
                                ]
                            }
                        ],
                        "temperature": 0.01,
                        "max_tokens": 196
                    }
                else:
                    data = {
                        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_1_base64}"}},
                                    {"type": "text", "text": user_prompt}
                                ]
                            }
                        ],
                        "temperature": 0.01,
                        "max_tokens": 196
                    }
                # print('len(content) = {}'.format(len(data['messages'][0]['content'])))

                response = requests.post(url, headers=headers, data=json.dumps(data))
                response = response.json()
                if 'choices' in response and len(response['choices']) > 0:
                    response = response['choices'][0]['message']['content']
                else:
                    print("Error: No choices in response")
                    print(response)
            else:
                raise ValueError("Invalid serve_choice. Choose from ['qwen_hf', 'qwen_vllm', 'llama3.2', 'qwen_72b']")

            print('response = {}'.format(response))

            # save response to a jsonl file along with ex_id, latent_idx, and species_label
            f1.write(json.dumps({'ex_id': ex_id, 'latent_idx': latent_idx, 'species_label': species_label, 'response': response}) + '\n')

            # flush the file
            f1.flush()
        
