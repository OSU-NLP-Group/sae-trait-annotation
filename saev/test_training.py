from . import config, training


def test_split_cfgs_on_single_key():
    cfgs = [config.Train(n_workers=12), config.Train(n_workers=16)]
    expected = [[config.Train(n_workers=12)], [config.Train(n_workers=16)]]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_on_single_key_with_multiple_per_key():
    cfgs = [
        config.Train(n_patches=12),
        config.Train(n_patches=16),
        config.Train(n_patches=16),
        config.Train(n_patches=16),
    ]
    expected = [
        [config.Train(n_patches=12)],
        [
            config.Train(n_patches=16),
            config.Train(n_patches=16),
            config.Train(n_patches=16),
        ],
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_on_multiple_keys_with_multiple_per_key():
    cfgs = [
        config.Train(n_patches=12, track=False),
        config.Train(n_patches=12, track=True),
        config.Train(n_patches=16, track=True),
        config.Train(n_patches=16, track=True),
        config.Train(n_patches=16, track=False),
    ]
    expected = [
        [config.Train(n_patches=12, track=False)],
        [config.Train(n_patches=12, track=True)],
        [
            config.Train(n_patches=16, track=True),
            config.Train(n_patches=16, track=True),
        ],
        [config.Train(n_patches=16, track=False)],
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_no_bad_keys():
    cfgs = [
        config.Train(n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=1e-4)),
        config.Train(n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=2e-4)),
        config.Train(n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=3e-4)),
        config.Train(n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=4e-4)),
        config.Train(n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=5e-4)),
    ]
    expected = [
        [
            config.Train(
                n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=1e-4)
            ),
            config.Train(
                n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=2e-4)
            ),
            config.Train(
                n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=3e-4)
            ),
            config.Train(
                n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=4e-4)
            ),
            config.Train(
                n_patches=12, sae=config.SparseAutoencoder(sparsity_coeff=5e-4)
            ),
        ]
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected
