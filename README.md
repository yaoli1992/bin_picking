# BinPicking


http://binpickingcad.blogspot.pt/

Conversão CAD par PointCloudData:

SolidWorks -(stl)-> Blender -(ply)-> mesh_to_pointcloud -(pcd)-> "model.pcd"


Diagrama de algoritmos para reconhecimento:



			"model.pcd"	"scene.pcd"

				\	 /
	            Compute Normals (NormalEstimationOMP)
				|	|
		    Compute Keypoints (ISSKeypoint3D)
				|	|
		    Compute Descriptor (SHOTEstimationOMP)
				|	|
		    Find Model-Scene Correspondences (KdtreeFLANN)
				    |
				Clustering (Houg3DGrouping or GeometricConsistencyGrouping)


















