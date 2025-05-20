#=
        Author: Rodrigo Kiefe

    Summary:
    A finite element implementation in Julia, demonstrating the calculation of
    the stiffness matrix and how it can be done with C++ instead, called within
    the Julia environment.

    Description:
    This repository aims to be a foundation. Something users can build on, to create 
    ground up implementations of the finite element method without compromising performance.
    This code base also demonstrates interoperability with C++ and high performance packages
    such as Eigen. As an example, this code shows how to calculate the dense, local stiffness matrix
    in C++, called from the Julia environment, with significant performance gains.
    Two examples are available: 1) Task intensive, called once.
                                2) Low compute time, called millions of times.
    In both scenarios, Julia benefits from calling the C++ variants of the same functions.
=#

# Gmsh powers the mesh generation and volume handling
using Gmsh
include("gmsh_wrapper.jl")  # FEMjl functions to operate Gmsh and
                            # output the mesh in a more familiar format

# Import FEMjl interpolation methods
include("Interpolation.jl")
import .interp as InP

# Required for solving matrix equations and exploit the sparsity of the matrices
using LinearAlgebra, SparseArrays

# For plots
using GLMakie


# Main function | Generates a geometry and mesh
function main(meshSize=0,localSize=0,showGmsh=false,saveMesh=false)
    #=
        An example of creating your own model with simple shapes
    =#
    
    # Create a geometry
    gmsh.initialize()

    # >> Model
    L::Vector{Float64} = [512,128,30]

    # Create an empty container
    # container = addCuboid([0,0,0],[4,4,4])
    container = addSphere([0,0,0],5*maximum(L))

    # Get how many surfaces compose the bounding shell
    temp = gmsh.model.getEntities(2)            # Get all surfaces of current model
    bounding_shell_n_surfaces = 1:length(temp)    # Get the number of surfaces in the bounding shell

    # List of cells inside the container
    cells = []

    # Add another object inside the container
    
    # addSphere([0,0,0],0.5,cells,true)
    addCuboid([0,0,0],L,cells,true)

    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, container)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    container = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,saveMesh)
    
    # Get bounding shell surface id
    shell_id = gmsh.model.getAdjacencies(3, container)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[bounding_shell_n_surfaces] # All other, are interior surfaces

    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))

    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)
end # end of main

# Example of handling cad files
function geometryFromCAD(meshSize=0,localSize=0,showGmsh=false,saveMesh=false)
    #=
        Imports a cad file, creates a bounding shell and a 3D mesh
    =#

    # Create a geometry
    gmsh.initialize()
    
    # List of cells inside the container
    cells = []

    # >> Model
    container = importCAD("STEP_Models/cube.step",cells)

    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, container)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    container = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,saveMesh)
    
    # Get bounding shell surface id
    shell_id = gmsh.model.getAdjacencies(3, container)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[1] # Using importCAD, the bounding shell is always a sphere

    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))

    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)
end

function NoBoundingShell(showGmsh)
    # Create a geometry
    gmsh.initialize()

    container = addSphere([0,0,0],1)
    container = addSphere([3,0,0],1)

    # List of cells inside the container
    cells = []
    
    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, container)], cells)
    gmsh.model.occ.synchronize()

    # Generate Mesh
    mesh = Mesh(cells,0,0,false)
    
    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))

    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)
end


meshSize = 500
localSize = 5
showGmsh = false
saveMesh = true

main(meshSize,localSize,showGmsh,saveMesh)
# geometryFromCAD()
# NoBoundingShell(showGmsh)