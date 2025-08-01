from Methods.algo import *
from DataProvider.Loaders import *
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="Config")
def main(config: DictConfig) -> None:
    domain_list, dataset_Loader = get_datsets(config, device = config.device)
    for target_domain in domain_list:
        print(target_domain)
        source_loaders, target_loader = dataset_Loader(config, target_domain)
        model = Model(config).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay = config.weight_decay)
        train(model, config.device, optimizer, source_loaders, target_loader, config)

if __name__ == "__main__":
    main()