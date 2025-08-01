def get_datsets(config, device = 'cuda'):
    if config.dataset == 'LENDB':
        from .LENDB_Loader import LENDB_dataPrep
        return list(range(config.n_domains)), LENDB_dataPrep(config).get_lendb_Loaders
    elif config.dataset == 'REFIT':
        from .REFIT_Loader import load_refit_Loaders, APPLIANCES_HOUSE
        return APPLIANCES_HOUSE[config.appliance], load_refit_Loaders
    elif config.dataset == 'PRSA':
        from .PRSA_Loader import PRSA_dataPrep
        return list(range(config.n_domains)), PRSA_dataPrep(config).get_Loaders
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")