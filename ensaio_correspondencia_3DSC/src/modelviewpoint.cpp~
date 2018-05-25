#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>

#include <pcl/features/normal_3d_omp.h>   //法线
#include <pcl/features/shot_omp.h>

#include <pcl/features/3dsc.h>     //3dsc
#include <pcl/features/board.h>
#include <pcl/features/feature.h>

//#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/uniform_sampling.h>
//#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

// ensaio de acerto de resolução
#include <iostream>


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>

#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <sstream>
#include <pcl/visualization/cloud_viewer.h>
// tempo
#include <ctime>
//conversão integer to string
#include <boost/lexical_cast.hpp>

/*
Eigen::Affine3f A;
Eigen::Matrix4f M;
M = A.matrix();
A = M;   // assume that M.row(3) == [0 0 0 1]
*/


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
/*
  Eigen::Affine3f f;
  f>>"-0.06,0.04,0.04";
*/
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(-0.06f, 0.4f,0.4f);  

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0,sensorPose);
  viewer->initCameraParameters ();
  return (viewer);
}




typedef pcl::PointXYZ PointType;
typedef pcl::PointWithViewpoint PointTypeViewPoint;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ShapeContext1980 DescriptorType1;
//typedef pcl::ReferenaceFrame RFType;

std::string model_filename_;


double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;

  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
 //  pcl::search::KdTree<PointType> tree;
    pcl::KdTreeFLANN<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))//判断点的x的值是否位无效点也就是NAN点   如果是无效点就不会计算
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
  //int pcl::search::Search<PointT>::nearSearch();是计算给定点的k个邻域的点
   //i 代表索引点，2是寻找周围的点树（其中第一个点是它本身）indices寻找到的周围点的索引sqr_distances与相邻点的平方距离
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);  //返回寻找到的点数
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}


int main(int argc,char** argv)
{

  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 1)
  {
    std::cout << "Filenames missing.\n";
    std::cout <<"you  shoud input a cloud point with .pcd file"<<std::endl;
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];

  pcl::PointCloud<PointTypeViewPoint>::Ptr modelviewpoint(new pcl::PointCloud<PointTypeViewPoint> ());
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType> ()); 
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr model_keypoints_normals (new pcl::PointCloud<NormalType> ());

  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());

  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());

 if(pcl::io::loadPCDFile (model_filename_, *model)<0)
   {
  std::cout<<"Error loading model point ."<<std::endl;
  return (-1);
  }
 
 if(pcl::io::loadPCDFile (scene_filename_,*scene)<0)
   {
   std::cout<<"Error laoding scene cloud. "<<std::endl;
   return (-1);
  }
/*
  float vpx = modelviewpoint->points[0].vp_x;
   std::cout<<"vpx : "<<vpx <<std::endl;
  float vpy = modelviewpoint->points[0].vp_y;
   std::cout<<"vpy : "<<vpy <<std::endl;
  float vpz = modelviewpoint->points[0].vp_z;
     std::cout<<"vpz : "<<vpz <<std::endl;

  pcl::copyPointCloud<PointTypeViewPoint,PointType>(*modelviewpoint, *model);
  */
  std::cout<<"before the uniform_samping the points has: "<<model->points.size()<<std::endl;
  double res_model = computeCloudResolution(model);
  std::cout << "Model resolution:  " << res_model << std::endl;

  //compute Normal
 pcl::NormalEstimationOMP<PointType,NormalType> norm_est;
 norm_est.setKSearch(10);
// norm_est.setRadiusSearch(0.002)
 norm_est.setInputCloud(model);
 //norm_est.setViewPoint(vpx,vpy,vpz);
 norm_est.compute(*model_normals);

 norm_est.setInputCloud(scene);
 norm_est.compute (*scene_normal);

//Downsample cloud to extraxt keypoints
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(model);
  uniform_sampling.setRadiusSearch(0.01*res_model);
  //pcl::PointCloud<int> keypointIndices1;
  uniform_sampling.filter(*model_keypoints);
std::cout<<"after the uniform_samping the mdoel keypoint points has: "<<model_keypoints->points.size()<<std::endl;
  
  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch(0.03*res_model);
  uniform_sampling.filter(*scene_keypoints);
std::cout<<"after the uniform_samping the scene keypoint points has: "<<scene_keypoints->points.size()<<std::endl;

//compute Decsriptor for keypoint
 pcl::SHOTEstimationOMP<PointType,NormalType,DescriptorType> descr_est;
 descr_est.setRadiusSearch(0.02*res_model);
  
 descr_est.setInputCloud(model_keypoints);
 descr_est.setInputNormals(model_normals);
 descr_est.setSearchSurface (model);
 descr_est.compute(*model_descriptor);
 
 descr_est.setInputCloud(scene_keypoints);
 descr_est.setInputNormals(scene_normals);
 descr_est.setSearchSurface(scene);
 descr_esr.compute(*scene_descriptors);


//find model   and  scene correspondence with kdtree
 pcl::CorrespondencePtr model_scene_corrs(new pcl::Correspondence());
 pcl::KdtreeFLANN<DescriptorType> match_search;
 match_search.setInputCloud(model_descriptors);

for (size_t i=0;i<scene_descriptors->size ();++i)
 {
  std::vector<int> neigh_indices(1);
  std::vector<float> neigh_sqr_dists (1);
   if(!pcl_isfinite (scene_desciptor->at (i).descriptors[0]))
  {
   continue;
  }
   
  int found_neighs = match_search.nearestKSearch (scene_descriptors-> at(i),1,neigh_indices,neigh_sqr_dists);
  if(found_neighs = 1&& neigh_sqr_dists[0]< 0.25)
    {
    pcl::Correspondence corr(neigh_indices[0],static_cast<int>(i),neigh_sqr_dists[0]);
  model_scene_corrs->push_back(corr);
   } 
 }
std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

//actual Clustering
 std::vector<Eigen::Matrix4f,EIgen::aligned_allocator<EIgen::Matrix4f>> rototranslations;
 std::vector<pcl::Correspondences> clustered_corrs;

 //use Hough  compute (keypoints) reference Frame only for Hough

pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

pcl::BOARDLocalReferenceFrameEstimation<PointType,NormalType,RFType> rf_est;
 rf_est.setFindHoles(true);
 rf_est.setRadiusSearch (0.015* res_model);
 
 rf_est.setInputCloud(model_keypoints);
 rf_est.setInputNormals (model_normal);
 rf_est.setSearchSurface (model);
 rf_est.compute (*model_rf);

 rf_est.setInputCloud(scene_keypoints);
 rf_est.setInputNormals (scene_normal);
 rf_est.setSearchSurface (scene);
 rf_est.compute (*scene_rf);

//clustering
pcl::Hough3DGrouping<PointType,PointType,RFType,RFType> clusterer;
clusterer.setHoughBinSize(0.01*res_model);
clusterer.setHoughThreshold(5.0);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);

   //output result
std::cout<<"MOdel instance found: " <<rototranslations.size()<<std::endl;
 for(size_t i =0; i<rototranslations.size();++i)
 {
  std::cout<<" \n  Intance "<< i+1 <<" : " std::endl;
  std::cout<<"        Correspondence belonging to this instance: "<<clustered_corrs[i].size() <<std::endl;

//print the rotation matrix and translation vector
  Eigen::Matrix3f rotation = rototranslation[i].block<3,3>(0,0);
  Eigen::Vector3f translation = rototranslations[i].block<3,1>(0,3);
    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

  }



 //pcl::copyPointCloud(*model,keypointIndices1.points,*model_keypoints);
//*******************************************************************************//

////#include <pcl/visualization/cloud_viewer.h>////////////////////////

    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    //blocks until the cloud is actually rendered
    viewer.showCloud(model_keypoints);
/*
  pcl::PCDWriter writer;
 writer.write("cloud_out_vpxyz.pcd",*model);
*/
    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer
/*    
    //This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);
    
    //This will get called once per visualization iteration
    viewer.runOnVisualizationThread (viewerPsycho);
*/
    

    while (!viewer.wasStopped ())
    {
    //you can also do cool processing here
    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //and you should guard against race conditions yourself...

    }
/*
 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1;
   viewer1 = simpleVis(model);
   viewer1->spin(); 
    return 0;
*/
//****************************************************************************//
//
/*
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = simpleVis(model);
  viewer->spin();
   return 0;
*/
}





















