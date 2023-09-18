#ifndef SETPCPP_HPP
#define SETPCPP_HPP

#include <pybind11/pybind11.h>
#include <euclidean_cluster.h>


#include "euclidean_cluster.h"

py::array_t<double> euclidean_cluster(py::array_t<double> data)
{
    // Read in the cloud data
    //pcl::PCDReader reader;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    //reader.read ("table_scene_lms400.pcd", *cloud);
    auto data_r = data.mutable_unchecked<2>();
    int no_points;
    no_points = data_r.shape(0);
    cloud->width    = no_points;
    cloud->height   = 1;
    cloud->is_dense = true;
    cloud->points.resize ((cloud->width) * (cloud->height));

    for(int i=0;i<no_points;i++)
    {
        cloud->points[i].x = data_r(i,0);
        cloud->points[i].y = data_r(i,1);
        cloud->points[i].z = data_r(i,2);
    }

    //std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
    //std::cout << "PointCloud before filtering has: " << no_points << " data points." << std::endl; //*

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.0001f, 0.0001f, 0.0001f);
    vg.filter (*cloud_filtered);
    //std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (5);
    seg.setDistanceThreshold (0.00001);

    int k=0;
    int cloud_sp = 0;
    int i=0, nr_points = (int) cloud_filtered->points.size ();
    if(1)
    {
        while (cloud_filtered->points.size () > 0.5 * nr_points)  //0.3
        {
            // Segment the largest planar component from the remaining cloud
            seg.setInputCloud (cloud_filtered);
            seg.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0 )
            {    
                
                //std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                break;
		
            }

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud (cloud_filtered);
            extract.setIndices (inliers);
            extract.setNegative (false);

            // Get the points associated with the planar surface
            extract.filter (*cloud_plane);
            //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

            // Remove the planar inliers, extract the rest
            extract.setNegative (true); //true
            extract.filter (*cloud_f);
            *cloud_filtered = *cloud_f;
            if (cloud_filtered->points.size () == 0 )
            {
                cloud_sp = 1;
                break;
            }
        }
    }

    else if(cloud_sp == 1)
    {
        auto ret_arr = py::array_t<double>(1);	
        auto n_ret = ret_arr.mutable_unchecked<>();
        n_ret(0) = -1;
	return ret_arr;
    }
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.05); // 1cm
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (30000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    int j = 0;
    int size_cl = cluster_indices.size();
    //cout<<size_cl;
    //cout<<"here";
    //py::array_t<double>* ret_arr = new py::array_t<double>[100];
    //py::array_t<double>* ret_r = ret_arr.mutable_unchecked<>();
    auto ret_arr = py::array_t<double>((size_cl+1)*500*3);	
    auto n_ret = ret_arr.mutable_unchecked<>();
    std::vector<pcl::PointIndices>::const_iterator it;
    k=0;

    for (it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        //std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        //std::stringstream ss;
        //ss << "cloud_cluster_" << j << ".pcd";
        //writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*

        for (int i=0;i<cloud_cluster->points.size()&&i<500;i++)
        {
            n_ret(k+i*3+0) = cloud_cluster->points[i].x;
            n_ret(k+i*3+1) = cloud_cluster->points[i].y;
            n_ret(k+i*3+2) = cloud_cluster->points[i].z;
        }
        for (int i=cloud_cluster->points.size();i<500;i++)
        {
            n_ret(k+i*3+0) = 0;
            n_ret(k+i*3+1) = 0;
            n_ret(k+i*3+2) = 0;
        }
        //cout<<n_ret(k+i*3+0)<<" "<<n_ret(k+i*3+1)<<" "<<n_ret(k+i*3+2);
        //cout<<"\n"<<j<<"\t";

        j++;
        k = j*500*3;
    }
    return ret_arr;
    //return data;
}


std::vector<float> enforceSmoothness(std::vector<float> data, float smoothness) {
    // Since this is pass by value, the data is already copied. Hence, smoothing can be done in-place.
    for(int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            if (data[i] > data[j])
                data[i] = min(data[i], data[j] + smoothness * abs(float(i - j)));
        }
    }
    return data;
}

PYBIND11_MODULE(setcpp, m) {
    m.def("euclidean_cluster", &euclidean_cluster, "Segmentation with Euclidean_cluster");
    m.def("enforce_smoothness", &enforceSmoothness, "Enforce smoothness of ranges");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}

#endif
