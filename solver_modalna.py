# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def read_input(inpFileName):
    print('Read input file')
    inputFile = open("Input_beam_v2.txt",'r')
    lines = inputFile.readlines()
    inputFile.close()
    Materialprops =[]
    Nodes=[]
    Elements=[]
    Boundary_dof=[]
    state = 0
    for line in lines:
        line = line.strip()
        if line[0] == '*':
            state = 0
        if line == '*Material_properties':
            state = 1
            continue
        if line == '*Node':
            state = 2
            continue
        if line == '*Element':
            state = 3
            continue
        if line == '*Boundary_dof':
            state = 4
            continue
        if state == 0:
            continue
        if state == 1: #read material properties
            values = line.split(",")
            Materialprops.append(float(values[1]))
            continue
        if state == 2:
            values  = line.split(",")
            Node_nr = int(values[0]) 
            x_cord = float(values[1])
            y_cord = float(values[2])
            Nodes.append([x_cord, y_cord])
            continue
        if state ==3:
            values  = line.split(",")
            Element_nr = int(values[0])
            n1 = int(values[1]) 
            n2 = int(values[2]) 
            n3 = int(values[3]) 
            n4 = int(values[4]) 
            Elements.append([n1, n2, n3, n4]) 
            continue
        if state == 4:
            values = line.split(",")
            nodeNr = int(values[0])   
            dof = int(values[1])
            displacement = float(values[2])
            Boundary_dof.append([nodeNr, dof, displacement])
            continue   
    print ("Material propertiess")
    print ("Density: ", Materialprops[0])
    print ("Young modulus: ", Materialprops[1])
    print ("Poisson ratio: ", Materialprops[2])
    print ("Nodes coordinates: ", Nodes)
    print ("Connectivity: ", Elements)
    print ("Prescribed displacements (node, dof, value): ", Boundary_dof)
    return [Materialprops, Nodes, Elements, Boundary_dof]


def plotmesh(coords,nr_nodes,connectivity,nr_elems):
    x = [];
    y = [];
    for i in range(len(coords)):
        print('i: ', i)
        print('   coords[i][1]):', coords[i][1]) 
        print('   coords[i][0]):', coords[i][0]) 
        x.append(coords[i][0])
        y.append(coords[i][1])
    plt.scatter(x,y)

#===SHAPE FUNCTION=====
def shapefun(xi):
#N(xi,eta)
#Input: 1x2,  Output: 1x4
        x,y = tuple(xi)
        N = [(1.0-x)*(1.0-y), (1.0+x)*(1.0-y), (1.0+x)*(1.0+y), (1.0-x)*(1.0+y)]
        return 0.25*np.array(N)
#===ELEMENT MASS MATRIX=====
def gradshapefun(xi):
#dN(xi,eta)/dxi 
#Input: 1x2,  Output: 2x4
	x,y = tuple(xi)
	dN = [[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],
		  [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]]
	return 0.25*np.array(dN)

#===ELEMENT MASS MATRIX=====
def element_massfun(coord,density):
#coord - nodes coordinates    
#density == materialprops[0]
   npoints = 4; #nr of integration points
#q4 - coordinates of gauss points
   q4 = np.array([[-0.5773502692,-0.5773502692],[0.5773502692,-0.5773502692],[-0.5773502692,0.5773502692],[0.5773502692,0.5773502692]])
   element_mass = np.zeros((8,8))	#element mass matrix is 8x8
   for q in q4:			#for each Gauss point
        N = shapefun(q)    
        dN = gradshapefun(q)
        J = np.dot(dN, coord).T # J is 2x2
        for a in range(0,4):
            for b in range(0,4):
                for i in range(0,2):
                    row = 2 * (a-1) + i
                    col = 2 * (b-1) + i
                    element_mass[row][col] += density * N[a] *N[b] * np.linalg.det(J)
        
        for a in range(0,8):
            for b in range(0,8):
               if (a != b):
                 element_mass[a][a] = element_mass[a][a] + element_mass[a][b];
                 element_mass[a][b] = 0.;                       
   return element_mass
   
   
#===GLOBAL MASS MATRIX=====
def globalmass(nr_nodes,nr_elems,coords,connectivity,density):
#global mass for quad elements
    M = np.zeros((2*nr_nodes,2*nr_nodes));
    lmncoord = np.zeros((4,2));
#Loop over all the elements
    for k, element_connectivity in enumerate(connectivity):
       for a in range(len(element_connectivity)):  #Extract coords of nodes for the current element
           lmncoord[a][0] = coords[element_connectivity[a]-1][0];
           lmncoord[a][1] = coords[element_connectivity[a]-1][1]
       element_mass = element_massfun(lmncoord,density);
       for c in range(0,4): 
           for i in range (0,2):
               for b in range(0,4):
                   for n in range(0,2):
                    rw = 2*(element_connectivity[c]-1)+i;
                    cl = 2*(element_connectivity[b]-1)+n;
                    M[rw][cl] = M[rw][cl] + element_mass[2*(c-1)+i][2*(b-1)+n];
    return M;     



#===ELEMENT STIFFNESS MATRIX===
def elementstiff(coord,materialprops):
#coord - nodes coordinates      
    q4 = np.array([[-0.5773502692,-0.5773502692],[0.5773502692,-0.5773502692],[-0.5773502692,0.5773502692],[0.5773502692,0.5773502692]])
    E = 26.0
    v = 0.3
    C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0], [v, 1.0-v, 0.0], [0.0, 0.0, 0.5-v]])
    B = np.zeros((3,8))
    k_element = np.zeros((8,8));
    for q in q4:
        N = shapefun(q)    
        dN = gradshapefun(q)       
        J = np.dot(dN, coord).T # J is 2x2
        M = np.dot(np.linalg.inv(J), dN) 
        B[0,0::2] = M[0,:]
        B[1,1::2] = M[1,:]
        B[2,0::2] = M[1,:]
        B[2,1::2] = M[0,:]
        k_element += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)        
    return k_element;             

    
#===Global stiffness matrix===
def globalstiffness(nr_nodes,nr_elems,coords,connectivity,density):
    K = np.zeros((2*nr_nodes, 2*nr_nodes)) #global stiffness matrix
    lmncoord = np.zeros((4,2));
    for k, element_connectivity in enumerate(connectivity):
        for a in range(len(element_connectivity)):  
           lmncoord[a][0] = coords[element_connectivity[a]-1][0];
           lmncoord[a][1] = coords[element_connectivity[a]-1][1]
        element_stiffness = elementstiff(lmncoord,density);
        for i,I in enumerate(element_connectivity):
           for j,J in enumerate(element_connectivity):
               K[2*(I-1),2*(J-1)]     += element_stiffness[2*i,2*j]
               K[2*(I-1)+1,2*(J-1)]   += element_stiffness[2*i+1,2*j]
               K[2*(I-1)+1,2*(J-1)+1] += element_stiffness[2*i+1,2*j+1]
               K[2*(I-1),2*(J-1)+1]   += element_stiffness[2*i,2*j+1]     
    return K; 


def main():
    Materialprops = [] 
    Nodes = [] 
    Elements = [] 
    Boundary_dof = []
    input_file = 'Input_beam_v2.txt'
    
    [Materialprops, Nodes, Elements, Boundary_dof] = read_input(input_file)
 
    print('Number of nodes:', len(Nodes))
    print('Number of elements:', len(Elements))
    print('Number of displacement boundary conditions:', len(Boundary_dof))  
 
        

    f, (ax1) = plt.subplots(figsize=(10,10))
    ax1.set_xlim([0, 20])
    ax1.set_ylim([-5, 5])
    ax1.set_aspect('equal')
    patches1 = []
    colors1 = np.zeros((len(Elements)))
    for i in range(len(Elements)):
        n1 = Elements[i][0]
        n2 = Elements[i][1]
        n3 = Elements[i][2]
        n4 = Elements[i][3]
        polygon = Polygon([Nodes[n1-1],Nodes[n2-1],Nodes[n3-1],Nodes[n4-1]], closed=True)
        patches1.append(polygon)
        colors1[i] = 1.0
    p1 = PatchCollection(patches1)
    p1.set_array(colors1)
    ax1.add_collection(p1)
    plt.colorbar(p1,ax=ax1)
    

    M = globalmass(len(Nodes),len(Elements),Nodes,Elements ,Materialprops[0])

    K = globalstiffness(len(Nodes),len(Elements),Nodes,Elements ,Materialprops[0])
    
    
    F = np.zeros((2*len(Nodes)))
#apply prescribed displacements  
    for i in range(len(Boundary_dof)): 
        nn  = Boundary_dof[i][0]
        dof = Boundary_dof[i][1]
        val = Boundary_dof[i][2]
        j = 2*nn-1
        if dof == 1: j = j - 1
        K[j,:] = 0.0
        K[:,j] = 0.0
        M[j,:] = 0.;
        M[:,j] = 0.;
        M[j,j] = 1.0
        F[j] = val

    
 
    rootM = np.sqrt(M);
    inverserootM = np.linalg.inv(rootM);
    H = np.dot(np.dot(inverserootM, K), inverserootM)
 
    u, s, vh = np.linalg.svd(H, full_matrices = True)

    r = np.zeros((2*len(Nodes)));
    for mode in range(0,4): 
       for i in range(0, 2*len(Nodes)):
         r[i] = u[i][2*len(Nodes) - 4 -mode]
       disp_test = np.dot(inverserootM,r);

       defcoords = np.zeros((len(Nodes),2))
       scale = 3.;
       for z in range(0,len(Nodes)):
           for kk in range(0,2):
               defcoords[z][kk] = Nodes[z][kk] + scale*disp_test[2*(z-1)+kk];
               
       f, (ax1) = plt.subplots(figsize=(10,10))
       ax1.set_xlim([0, 20])
       ax1.set_ylim([-5, 5])
       ax1.set_aspect('equal')
       patches1 = []
       colors1 = np.zeros((len(Elements)))
       for i in range(len(Elements)):
          n1 = Elements[i][0]
          n2 = Elements[i][1]
          n3 = Elements[i][2]
          n4 = Elements[i][3]
          polygon = Polygon([defcoords[n1-1],defcoords[n2-1],defcoords[n3-1],defcoords[n4-1]], closed=True)
          patches1.append(polygon)
          colors1[i] = 1.0
          p1 = PatchCollection(patches1)
          p1.set_array(colors1)
          ax1.add_collection(p1)
          plt.savefig('Mode_{}.png'.format(mode+1), format="PNG")

main()    
		