# Ordinal Aligned Domain Generalization for Sensor-based Time Series Regression

## Abstract
Time series data powers sensor systems in health, cities, and beyond, demanding robust analysis for real-world impact. While deep learning models excel in this field, their performance degrades in new environments due to data distribution shifts. Domain generalization (DG) aims to enhance model performance in new environments, but current methods primarily focus on discrete data, assuming a discrete, fixed label space, and addressing distribution shifts by extracting common features from inputs across all source domains. However, sensor-based tasks involve real-valued data with diverse input and label spaces. Existing approaches overlook the continuity between data and labels, mapping input data with similar labels to scattered feature spaces, making models susceptible to distribution shifts. Additionally, variations in the label space cause predictive features to change across domains, complicating the identification of stable, generalizable features. 
This work introduces a new DG framework tailored for sensor-based tasks, operating without access to target domain data or post-deployment adjustments. 
Our approach learns Ordinal-Aligned Task-Specific (OATS) features that capture stable relationships between continuous labels and input features while maintaining domain independency under input and label space shift. 
This enables the model to make accurate predictions across unseen domains and continuous label spaces.

## Overview

- **DataProvider**: Handles data preprocessing and loading routines.
- **Methods**: Implements core algorithms and supporting utility functions.
- **Data**: Directory containing processed datasets.
- **Main.py**: The main script for running the experiments.

## Run
> python main.py --config-name=\$\{dataset\_name} data_dir=\$\{your\_dir\}

## Data License
LENDB[1], REFIT[2] and PRSA[3] datasets are available under the Creative Commons Attribution 4.0 International License, permitting use, distribution, and reproduction in any medium, provided the original work is properly credited. Biovid[4] is available for non-commercial research only.

## Reference
[1] Magrini, F., Jozinović, D., Cammarano, F., Michelini, A. and Boschi, L., 2020. Local earthquakes detection: A benchmark dataset of 3-component seismograms built on a global scale. Artificial Intelligence in Geosciences, 1, pp.1-10.  
[2] Zhang, S., Guo, B., Dong, A., He, J., Xu, Z. and Chen, S.X., 2017. Cautionary tales on air-quality improvement in Beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205), p.20170457.  
[3] Murray, D., Stankovic, L. and Stankovic, V., 2017. An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. Scientific data, 4(1), pp.1-12.  
[4] Walter, S., Al-Hamadi, A., Nienaber, S. and Dinges, L., 2013. The BioVid Heat Pain Database: A multimodal dataset for the study of pain. University of Bamberg, Germany.
