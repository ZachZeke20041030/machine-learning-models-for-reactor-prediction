This model aims to predict key nuclear reactor outputs:
- Outlet Temperature  
- Thermal Power  
- Gross Electrical Power  

using supervised machine learning models.

## Models
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Regression
- Custom MLP
- Stacking Ridge (meta-learning)
- PINN (Physics-Informed Neural Network, 4 hidden layers)

##Dataset
Categorical parameters:
- Reactor type
Input parameters:
- Fuel enrichment
- Core height
- Core diameter
- Fuel linear heat generation
- Fuel assemblies
- Control rod assemblies
- Coolant pressure
Output parameters:
- Outlet Temperature
- Thermal Power
- Gross Electrical Power

Dataset Source:
https://www.iaea.org/publications/15752/operating-experience-with-nuclear-power-stations-in-member-states-2024-edition

## Requirements
See `requirements.txt`
