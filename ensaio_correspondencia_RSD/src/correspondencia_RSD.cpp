#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/rsd.h>
#include <pcl/features/board.h>

//#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/uniform_sampling.h>
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
#include <pcl/filters/voxel_grid.h>

#include <sstream>

// tempo
#include <ctime>
//conversão integer to string
#include <boost/lexical_cast.hpp>


typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;

typedef pcl::PointWithViewpoint PointTypeViewPoint;

typedef pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr  DescriptorType(new pcl::PointCloud<pcl::PrincipalRadiiRSD>());

std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool show_normals_(false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);

//Keypoints parameters
bool use_uniformsampling(true);// default
bool use_iss(false);
bool use_harris3d(false);

//Features parameters
bool use_shot(true);//default
bool use_pfh(false);
bool use_usd(false);

float model_ss_ (0.005f);
float scene_ss_ (0.005f);

float rf_rad_ (0.05f);
float descr_rad_ (0.05f); //:q:;:::)jijo

float cg_size_ (0.08f);
float cg_thresh_ (1.0f);

float Lim_dist_ (20000000.0f);

// Cronometro
std::clock_t start;
std::clock_t end;

double diffclock(std::clock_t clock1,std::clock_t clock2)
{
  double diffticks = clock2 - clock1;
  double diffms = diffticks / (CLOCKS_PER_SEC / 1000 );
  return diffms;
}

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                            Usage Guide                                  *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -n:                     Show used Normals." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --Keypoints: keypoints downsample algorithm used (default UniformSampling)." << std::endl;
  std::cout << "                  ISS (Intrinsic Shape Signatures)" << std::endl;
  //std::cout << "                  Harris3D                        " << std::endl;
  std::cout << "     --Features:  descritor features algorithm used (default SHOT)." << std::endl;
  //std::cout << "                  PFH (Point Features Histogram)" << std::endl;
  //std::cout << "                  ISS (Intrinsic Shape Signatures)" << std::endl;
  //std::cout << "                  USD (Unique Shape Context)" << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];


  //Program visualization behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  if (pcl::console::find_switch (argc, argv, "-n"))
  {
    show_normals_ = true;
  }


  // selection of algorithm for clustering
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }
    else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }


  // Selection of algorithm for Keypoint extration
  std::string used_keypoints;
  if (pcl::console::parse_argument (argc, argv, "--keypoint", used_keypoints) != -1)
  {
    if (used_keypoints.compare ("UniformSample") == 0)
    {

    }
    else if (used_keypoints.compare ("ISS") == 0)
    {
      use_uniformsampling = false;
      use_iss = true;
    }
    else if (used_keypoints.compare ("Harris3D") == 0)
    {
      use_uniformsampling = false;
      use_harris3d = true;
    }
    else
    {
      std::cout << "Wrong Keypoint name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }


  // Selection of features
/*  std::string used_features;
  if (pcl::console::parse_argument (argc, argv, "--features", used_features) != -1)
  {
    if (used_algorithm.compare ("SHOT") == 0)
    {

    }
    else if (used_algorithm.compare ("PFH") == 0)
    {
        use_shot = false;
        use_pfh = true;
    }
    else if (used_algorithm.compare ("USD") == 0)
    {
        use_shot = false;
        use_usd = true;
    }
    else
    {
      std::cout << "Wrong Feature name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }*/

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

//***********************************************************************************************
//                                  Função Cálculo resolução nuvem
//***********************************************************************************************


double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;

  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
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



int
main (int argc, char *argv[])
{
  parseCommandLine (argc, argv);

  pcl::PointCloud<PointTypeViewPoint>::Ptr modelviewpoint (new pcl::PointCloud<PointTypeViewPoint> ());

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());

  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());

  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());

  pcl::PointCloud<NormalType>::Ptr model_keypoints_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_keypoints_normals (new pcl::PointCloud<NormalType> ());

  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

  //
  //  Load clouds
  //
  if (pcl::io::loadPCDFile (model_filename_, *modelviewpoint) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[1]);
    return (-1);
  }

  // Recolha dos view points para calculo das normais
  float vpx = modelviewpoint->points[0].vp_x;
  float vpy = modelviewpoint->points[0].vp_y;
  float vpz = modelviewpoint->points[0].vp_z;

  // Copia da nuvem para tipologia sem view points
  pcl::copyPointCloud<PointTypeViewPoint,PointType>(*modelviewpoint, *model);


  // see the resolution of the clouds
  start = std::clock();                                    // Debug time

  double res_scene = computeCloudResolution(scene);
  double res_model = computeCloudResolution(model);

  end = std::clock();                                       // Debug time
  std::cout << "Time computing clouds resolution:  " << diffclock(start, end) << " ms" << endl << endl;

  std::cout << "Scene resolution:  " << res_scene << std::endl;
  std::cout << "Model resolution:  " << res_model << std::endl;

  //
  //  Set up resolution invariance
  //
  if (use_cloud_resolution_)
  {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  }

  //
  //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (100);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);

  norm_est.setInputCloud (model);
  norm_est.setViewPoint(vpx,vpy,vpz);
  norm_est.compute (*model_normals);


//
//  Downsample Clouds to Extract keypoints
//
        if(use_uniformsampling)
        {
        pcl::UniformSampling<PointType> uniform_sampling;
        uniform_sampling.setInputCloud (model);
        uniform_sampling.setRadiusSearch (model_ss_);
        //uniform_sampling.filter (*model_keypoints);
        pcl::PointCloud<int> keypointIndices1;
        uniform_sampling.compute(keypointIndices1);

        pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints);
        pcl::copyPointCloud(*model_normals, keypointIndices1.points, *model_keypoints_normals);

        std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;


        uniform_sampling.setInputCloud (scene);
        uniform_sampling.setRadiusSearch (scene_ss_);
        //uniform_sampling.filter (*scene_keypoints);
        pcl::PointCloud<int> keypointIndices2;
        uniform_sampling.compute(keypointIndices2);

        pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints);
        pcl::copyPointCloud(*scene_normals, keypointIndices2.points, *scene_keypoints_normals);

        std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
        }




  //
  //  Downsample Clouds to Extract keypoints ////////////////////////////////////////////////
  //
        if(use_iss)
        {
        // ISS keypoint detector model.
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
        detector.setInputCloud(model);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_model(new pcl::search::KdTree<pcl::PointXYZ>);
        detector.setSearchMethod(kdtree_model);
        double resolution_model = computeCloudResolution(model);
        // Set the radius of the spherical neighborhood used to compute the scatter matrix.
        detector.setSalientRadius(6 * resolution_model);
        // Set the radius for the application of the non maxima supression algorithm.
        detector.setNonMaxRadius(4 * resolution_model);
        // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
        detector.setMinNeighbors(5);
        // Set the upper bound on the ratio between the second and the first eigenvalue.
        detector.setThreshold21(0.975);
        // Set the upper bound on the ratio between the third and the second eigenvalue.
        detector.setThreshold32(0.975);
        // Set the number of prpcessing threads to use. 0 sets it to automatic.
        detector.setNumberOfThreads(4);

        detector.setNormalRadius (4 * resolution_model);
        detector.setBorderRadius (1 * resolution_model);

        detector.compute(*model_keypoints);
        std::cout << "model_keypoints: " << model_keypoints->size () << std::endl;//debug



        // ISS keypoint detector scene.
        //pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
        detector.setInputCloud(scene);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_scene(new pcl::search::KdTree<pcl::PointXYZ>);
        detector.setSearchMethod(kdtree_scene);
        double resolution_scene = computeCloudResolution(scene);
        // Set the radius of the spherical neighborhood used to compute the scatter matrix.
        detector.setSalientRadius(6 * resolution_scene);
        // Set the radius for the application of the non maxima supression algorithm.
        detector.setNonMaxRadius(4 * resolution_scene);
        // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
        detector.setMinNeighbors(5);
        // Set the upper bound on the ratio between the second and the first eigenvalue.
        detector.setThreshold21(0.975);
        // Set the upper bound on the ratio between the third and the second eigenvalue.
        detector.setThreshold32(0.975);
        // Set the number of prpcessing threads to use. 0 sets it to automatic.
        detector.setNumberOfThreads(4);

        detector.setNormalRadius (4 * resolution_scene);
        detector.setBorderRadius (1 * resolution_scene);

        detector.compute(*scene_keypoints);
        std::cout << "scene_keypoints: " << scene_keypoints->size () << std::endl;//debug
        }
  //
  //  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //


  //
  //  Compute Descriptor for keypoints SHOT
  //
/*      if(use_shot)
        {
        pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
        descr_est.setRadiusSearch (descr_rad_);

        descr_est.setInputCloud (model_keypoints);
        descr_est.setInputNormals (model_normals);
        descr_est.setSearchSurface (model);
        descr_est.compute (*model_descriptors);

        descr_est.setInputCloud (scene_keypoints);
        descr_est.setInputNormals (scene_normals);
        descr_est.setSearchSurface (scene);
        descr_est.compute (*scene_descriptors);
        }
*/

  //
  //  Compute Descriptor for keypoints PFH
  //
/*      if(use_pfh)
        {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::PFHEstimation<PointType, NormalType, DescriptorType> descr_est;
        descr_est.setRadiusSearch (descr_rad_);

        descr_est.setInputCloud (model_keypoints);
        descr_est.setInputNormals (model_keypoints_normals);
        descr_est.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        descr_est.compute (*model_descriptors);

        descr_est.setInputCloud (scene_keypoints);
        descr_est.setInputNormals (scene_keypoints_normals);
        descr_est.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        descr_est.compute (*scene_descriptors);
        }
*/

  //
  //  Compute Descriptor for keypoints RSD
  //

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::RSDEstimation<PointType, NormalType, pcl::PrincipalRadiiRSD> descr_est;
        descr_est.setRadiusSearch (descr_rad_);
        descr_est.setPlaneRadius(0.1);

        descr_est.setInputCloud (model_keypoints);
        descr_est.setInputNormals (model_keypoints_normals);
        descr_est.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        descr_est.setSaveHistograms(false);
        descr_est.compute (*model_descriptors);



        descr_est.setInputCloud (scene_keypoints);
        descr_est.setInputNormals (scene_keypoints_normals);
        descr_est.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        descr_est.compute (*scene_descriptors);
        //}



  //
  //  Find Model-Scene Correspondences with KdTree for SHOT
  //
/*if(use_shot)
{
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  //  Para cada descritor de ponto chave da cena, encontra o mais proximo vizinho dentro dos descritores de pontos chave do modelo e junta os ao vector de correspondecias.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);

    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }

    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.30f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
}*/



  //
  //  Find Model-Scene Correspondences with KdTree for PFH
  //

//if(use_pfh)
//{
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  //  Para cada descritor de ponto chave da cena, encontra o mais proximo vizinho dentro dos descritores de pontos chave do modelo e junta os ao vector de correspondecias.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);

            /*  std::cout << " desccriptors " << i << ":" << scene_descriptors->at (i).histogram[0] <<
                                                   " " << scene_descriptors->at (i).histogram[1] <<
                                                   " " << scene_descriptors->at (i).histogram[2] <<
                                                   " " << scene_descriptors->at (i).histogram[3] <<
                                                   " " << scene_descriptors->at (i).histogram[4] <<
                                                   " " << scene_descriptors->at (i).histogram[5] <<
                                                   " " << scene_descriptors->at (i).histogram[6] <<
                                                   " " << scene_descriptors->at (i).histogram[7] <<
                                                   " " << scene_descriptors->at (i).histogram[8] <<
                                                   " " << scene_descriptors->at (i).histogram[9] <<
                                                   " " << scene_descriptors->at (i).histogram[10] <<
                                                   " " << scene_descriptors->at (i).histogram[11] <<
                                                   " " << scene_descriptors->at (i).histogram[12] <<
                                                   " " << scene_descriptors->at (i).histogram[13] <<
                                                   " " << scene_descriptors->at (i).histogram[14] <<
                                                   " " << scene_descriptors->at (i).histogram[15] <<
                                                   " " << scene_descriptors->at (i).histogram[16] <<
                                                   " " << scene_descriptors->at (i).histogram[17] <<
                                                   " " << scene_descriptors->at (i).histogram[18] <<
                                                   " " << scene_descriptors->at (i).histogram[19] <<
                                                   " " << scene_descriptors->at (i).histogram[20] <<
                                                   " " << scene_descriptors->at (i).histogram[21] <<
                                                   " " << scene_descriptors->at (i).histogram[22] <<
                                                   " " << scene_descriptors->at (i).histogram[23] <<
                                                   " " << scene_descriptors->at (i).histogram[24] <<
                                                   " " << scene_descriptors->at (i).histogram[25] <<
                                                   " " << scene_descriptors->at (i).histogram[26] <<
                                                   " " << scene_descriptors->at (i).histogram[27] <<
                                                   " " << scene_descriptors->at (i).histogram[28] <<
                                                   " " << scene_descriptors->at (i).histogram[29] <<
                                                   " " << scene_descriptors->at (i).histogram[30] <<
                                                   " " << scene_descriptors->at (i).histogram[31] <<
                                                   " " << scene_descriptors->at (i).histogram[32] <<
                                                   " " << scene_descriptors->at (i).histogram[33] <<  std::endl;*/


    if (!pcl_isfinite (scene_descriptors->at (i).histogram[0])) //skipping NaNs
    {
      continue;
    }

    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);

        //  std::cout << "neigh_sqr_dists: " << neigh_sqr_dists[0] << std::endl; //debugg

    if(found_neighs == 1 && neigh_sqr_dists[0] < Lim_dist_) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
//}









  //
  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  //  Using Hough3D
  if (use_hough_)
  {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
  }
  else // Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);
    std::cout << "G" << std::endl; //debug

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
    std::cout << "H" << std::endl; //debug
  }










  //
  //  Output results
  //
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }



  //
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler (scene, 255, 0, 255);
  viewer.addPointCloud (scene,scene_color_handler, "scene_cloud");

  std::string       id = "reference";
  viewer.addCoordinateSystem(0.1, id, 0);

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  if (show_correspondences_ || show_keypoints_)
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 0);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }




  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 128, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 128, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }





  if (show_normals_)
  {

  viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (scene, scene_normals, 1, 0.005, "normals");
  viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (off_scene_model, model_normals, 1, 0.005, "normal");

  }






  for (size_t i = 0; i < rototranslations.size (); i++)
  {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

    int inst = i;// + ((v + 1) * 10);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << inst;

    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

          // geração da matriz de rotação para os referenciais dos objectos detectados
          Eigen::Affine3f t;

          Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
          Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

          t.linear() = rotation;
          t.translation() = translation;

          //defenição dos id's para os vários referenciais
          std::string number;
          std::stringstream strstream;

          strstream << inst;
          strstream >> number;
          //std::string &id = i;


          viewer.addCoordinateSystem(0.05, t, number, 0);


    if (show_correspondences_)
    {
      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
        PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
      }
    }
  }



  //
  //Visualização dos descritores da cena
  //
  std::cout << "scene_descriptors size:" << scene_descriptors->size() << std::endl;
  for (size_t i = 0; i < scene_descriptors->size (); i+=10)
  {
    std::string number;
    std::string ValDes;

    std::ostringstream ss1;

        ss1 << std::fixed << std::setprecision(4);
        ss1 << i;

          number = boost::lexical_cast<std::string>(i);

          double textScale = 0.001;
          double         r = 1.0;
          double         g = 1.0;
          double         b = 1.0;
          const std::string & id = number;
          int            viewport = 1;

          for(size_t j = 0; j < 33; j++)
          {
            std::ostringstream ss;

                ss << std::fixed << std::setprecision(4);
                ss << scene_descriptors->at(i).histogram[j];

                if (j==0) {ValDes.append(ss1.str() + ": ");}


          ValDes.append(ss.str() + ", ");

          //ValDes.push_back();

          }

        //  std::cout << ValDes << std::endl;

          //std::cout << " desccriptor " << i << ":" << scene_descriptors->at (i).histogram[5] << std::endl;

          viewer.addText3D   (  number, scene_keypoints->points[i], textScale, r, g, b, id, viewport);
  }



  //
  //Visualização dos descritores do modelo
  //
  std::cout << "model_descriptors size:" << model_descriptors->size() << std::endl;
    for (size_t i = 0; i < model_descriptors->size (); i+=10)
    {
      std::string number;
      std::string ValDes;

      std::ostringstream ss2;

          ss2 << std::fixed << std::setprecision(4);
          ss2 << i;


            number = boost::lexical_cast<std::string>(i);

            double textScale = 0.001;
            double         r = 1.0;
            double         g = 1.0;
            double         b = 1.0;
            const std::string & id = number;
            int            viewport = 1;

            for(size_t j = 0; j < 33; j++)
            {
              std::ostringstream ss;

                  ss << std::fixed << std::setprecision(2);
                  ss << model_descriptors->at(i).histogram[j];

                  if (j==0) {ValDes.append(ss2.str() + ": ");}


            ValDes.append(ss.str() + ", ");

            //ValDes.push_back();

            }

          //  std::cout << ValDes << std::endl;

            //std::cout << " desccriptor " << i << ":" << scene_descriptors->at (i).histogram[5] << std::endl;

            viewer.addText3D   (  number, off_scene_model_keypoints->points[i], textScale, r, g, b, id, viewport);
    }
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}
