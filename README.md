# torque-ripple-compensation

Iterative learning control (ILC) and Q-learning can be used for reducing torque pulsations. This repository contains source code for both compensation methods. Additionally, a simple pulsation model is also provided. The model is needed when compensation performance is studied in simulations, as the disturbances must be modelled somehow.

## Experimenting with the Q-learning based method
Q-learning based compensator in use:  
[![Q-learning based method used for compensation](https://img.youtube.com/vi/ElfED9npK5o/0.jpg)](https://youtu.be/ElfED9npK5o)  
The speed ripple gets reduced when compensator is enabled and q-axis current amplitude increases due to applied compensation.