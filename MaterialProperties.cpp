/**
 * @file        MaterialProperties.cpp
 * @author      Jiri Jaros and Vojtech Nikl\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file with the medium properties
 *
 * @version     2017
 * @date        19 February 2015, 16:22 (created) \n
 *              28 March 2017, 12:02 (revised)
 *
 * @detail
 * This implementation file defines the class for material properties
 */

#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <hdf5.h>

#include "MaterialProperties.h"
#include "BasicRoutines.h"



using namespace std;


/**
 * Destructor.
 */
TMaterialProperties::~TMaterialProperties()
{
  _mm_free(domainMap);
  _mm_free(domainParams);
  _mm_free(initTemp);
} // end of TMaterialProperties
//------------------------------------------------------------------------------


///
/**
 * Load data from file
 * @param [in] fileName
 */
void TMaterialProperties::LoadMaterialData(const string fileName, bool loadData)
{
  hid_t file_id, dataset_id;

  try {
  // Open an existing file.
  file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) ios::failure("Cannot open input file");

  // Open the input file and load the domain size
  dataset_id = H5Dopen(file_id, "/EdgeSize", H5P_DEFAULT);
  if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset EdgeSize");
  H5Dread(dataset_id, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &edgeSize);
  H5Dclose(dataset_id);

  nGridPoints = edgeSize * edgeSize;
 
  // memory allocation based on the domain size
  domainMap    = (int  *) _mm_malloc(nGridPoints * sizeof(int),   DATA_ALIGNMENT);
  domainParams = (float*) _mm_malloc(nGridPoints * sizeof(float), DATA_ALIGNMENT);
  initTemp     = (float*) _mm_malloc(nGridPoints * sizeof(float), DATA_ALIGNMENT);

  if (!(domainMap && domainParams && initTemp)) throw std::bad_alloc();

  // Open Size and check dimensions
  dataset_id = H5Dopen(file_id, "/CoolerTemp", H5P_DEFAULT);
  if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset CoolerTemp");
  H5Dread(dataset_id, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &coolerTemp);
  H5Dclose(dataset_id);

  // Open S
  dataset_id = H5Dopen(file_id, "/HeaterTemp", H5P_DEFAULT);
  if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset HeaterTemp");
  H5Dread(dataset_id, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &heaterTemp);
  H5Dclose(dataset_id);

  if (loadData)
  {
    // Read domain map
    dataset_id = H5Dopen(file_id, "/DomainMap", H5P_DEFAULT);
    if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset DomainMap");
    H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, domainMap);
    H5Dclose(dataset_id);

    // Read Domain parameters.
    dataset_id = H5Dopen(file_id, "/DomainParameters", H5P_DEFAULT);
    if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset DomainParameters");
    H5Dread(dataset_id, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, domainParams);
    H5Dclose(dataset_id);

    // Read Initial Temperature
    dataset_id = H5Dopen(file_id, "/InitialTemperature", H5P_DEFAULT);
    if (dataset_id == H5I_INVALID_HID) throw ios::failure("Cannot open dataset InitialTemperature");
    H5Dread(dataset_id, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, initTemp);
    H5Dclose(dataset_id);
  }

  } catch (...) {
    fprintf(stderr, "Wrong input material file: %s\n", fileName.c_str());
    MPI_Abort(MPI_COMM_WORLD, -1); 
  }

  // Close the file.
  H5Fclose(file_id);
} // end of LoadMaterialData
//------------------------------------------------------------------------------
