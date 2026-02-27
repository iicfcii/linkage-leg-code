# Code for Multi-Degree-of-Freedom Planar Linkage Synthesis for Leg Designs

## Setup
Clone this repository and create the Python environment. An environment file for [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) is provided and can be used with the following command. 

```
conda env create -f environment.yml
```

The installation might take a while. Once finished, the conda environment can be activated with the following command. 

```
conda activate linkage-leg
```

## Examples
All examples should be run at the root directory of this repository. An Nvidia GPU is recommended to improve speed. This code also uses `torch.compile` for further and very significant speedup, which is not well supported on Windows. 

### Enumerate Topologies
This example enumerates all the possible topologies and save the intermediate and final files under the automatically created `logs` folder. This run will take a while. 

**A pre-generated file is already included in the repository, but this example is useful if the topology generation algorithm is to be modified.** 

```
python -m design.enum_topologies
```

The following command can be used to resume a previous enumeration. 

```
python -m examples.enum_topologies PATH/TO/FOLDER
```

### Filter and Plot Topologies
This example shows how to filter the topologies and plot some, which is useful for investigating the available topologies and narrowing down specific ones for dimensional synthesis. 

```
python -m examples.filter_and_plot_topologies
```

### One-DoF Leg
This example shows how to design a one-DoF leg for walking mechanisms. 

Test some randomly sampled designs for a topology. 

```
python -m examples.one_dof_leg t 0
```

Optimize the designs for a topology and save the design file in the `logs` folder. 

```
python -m examples.one_dof_leg o 0
```

Sweep through all the topologies. Optimize and save their respective designs. 

```
python -m examples.one_dof_leg s
```

Plot the designs in a design file.

```
python -m examples.one_dof_leg p PATH/TO/FILE
```

### Two-DoF Parllel Leg
This example shows how to design a two-DoF parllel leg with a fan-shaped workspace. Same commands are available. 

```
python -m examples.two_dof_parallel_leg t 0
```

```
python -m examples.two_dof_parallel_leg o 0
```

```
python -m examples.two_dof_parallel_leg s
```

```
python -m examples.two_dof_parallel_leg p PATH/TO/FILE
```

### Three-DoF Hybrid Leg
This example shows how to design a thre-DoF hybrid leg with an active ankle. Same commands are available. 

```
python -m examples.three_dof_hybrid_leg t 0
```

```
python -m examples.three_dof_hybrid_leg o 0
```

```
python -m examples.three_dof_hybrid_leg s
```

```
python -m examples.three_dof_hybrid_leg p PATH/TO/FILE
```
