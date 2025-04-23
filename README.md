# FEMjl
Make a high quality 3D mesh of your 3D model and get all the properties you need, easily accessible in a simple MESH() object, powered by Gmsh. Make a local refinement of your model, based on the volume ID, or just set a target mesh size.

Automatically create a bounding shell for your 3D model, simplifying magnetic field simulations. You can define the scale of your bounding shell directly in import phase of you .step file, or keep the default "5x larger". The local refinement is automatically set for every cell that isn't the container volume.

Both internal and bounding shell surfaces are preserved. You can access the surface ID of each surface triangle of your mesh directly.

You can make your own simple models with cuboids and spheres. Each cell ID you add is tracked for you. Just "addCuboid" or "addSphere" and you are set.

Automatically get the mesh element volumes, surface triangle normals and the area of each surface triangle.

The output mesh object is optimized for Finite-Element simulations. The main.jl includes an example of creating the stiffness matrix.

### Installation
To install, make sure you add Gmsh and include the gmsh_wrapper.jl. That's it.
![twoBalls](https://github.com/user-attachments/assets/3b9549ba-3968-40f1-94a4-5c21ce37ca9e)
