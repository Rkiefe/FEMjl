# FEMjl
Make a high quality 3D mesh of your 3D model and get all the properties you need, easily accessible in a simple MESH() object, powered by Gmsh. Make a local refinement of your model, based on the volume ID, or just set a target mesh size.

Automatically create a bounding shell for your 3D model, simplifying magnetic field simulations. You can define the scale of your bounding shell directly in import phase of you .step file, or keep the default "5x larger". The local refinement is automatically set for every cell that isn't the container volume.

You can import your geometry (and automatically create a bounding shell for open boundary problems) with
```
importCAD(file)
```

Or make your own geometry with cuboids
```
box = addCuboid(position_center,[W,D,H])
```
And/or spheres as
```
addSphere(position_center,sphere_radius,cells)
```
Where `cells` is an array of volume ID's inside the bounding shell (considering you have solids inside a defined space by a bounding shell, such as with open boundary problems). Each cell ID you add is tracked for you.
You can generate a mesh for your volume simply by
```
mesh = Mesh(cells,meshSize,localSize,saveMesh)
```

Both internal and bounding shell surfaces are preserved. You can access the surface ID of each surface triangle of your mesh directly.


Automatically get the mesh element volumes, surface triangle normals and the area of each surface triangle.

The output mesh object is optimized for Finite-Element simulations. The main.jl includes an example of creating the stiffness matrix.

### Installation
To install, go to your Julia repl and add Gmsh, LinearAlgebra and SparseArrays. That's it.
![twoBalls](https://github.com/user-attachments/assets/3b9549ba-3968-40f1-94a4-5c21ce37ca9e)

### Running C++ variants withing the Julia environment
Recently, this repository was updated to include an example of how you can add C++ functions to speed up calculations within Julia. The example demonstrates how to calculate the local, dense stiffness matrix in C++ within Julia.
To install, first make sure you have Eigen https://eigen.tuxfamily.org/index.php?title=Main_Page, update the FEMc.cpp in the `#include` to point to your Eigen directory and then compile the FEMc.cpp file as a shared library (`g++ -O3 -fPIC -shared -o FEMc.so FEMc.cpp`) and it is good to go.
