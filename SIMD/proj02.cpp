/**
 * @file        proj02.cpp
 * @author      Jiri Jaros, Radek Hrbacek, Filip Vaverka and Vojtech Nikl\n
 *              Faculty of Information Technology\n
 *              Brno University of Technology\n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Parallelisation of Heat Distribution Method in Heterogenous
 *              Media using MPI and OpenMP
 *
 * @version     2017
 * @date        10 April 2015, 10:22 (created)\n
 *              28 March 2017, 12:02 (revised)
 *
 * @detail
 * This is the main file of the project. Add all code here.
 */


#include <mpi.h>
#include <omp.h>

#include <string.h>
#include <string>
#include <cmath>
#include <iostream>

#include <hdf5.h>

#include <sstream>
#include <immintrin.h>

#include "MaterialProperties.h"
#include "BasicRoutines.h"

#include <vector>
#include <stdexcept>

using namespace std;



//----------------------------------------------------------------------------//
//---------------------------- Global variables ------------------------------//
//----------------------------------------------------------------------------//

/// Temperature data for sequential version.
float *seqResult = NULL;
/// Temperature data for parallel method.
float *parResult = NULL;

/// Parameters of the simulation
TParameters parameters;

/// Material properties
TMaterialProperties materialProperties;


//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Sequential implementation of the Heat distribution
void SequentialHeatDistribution(float *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters &parameters,
                                string outputFileName);

/// Parallel Implementation of the Heat distribution (Non-overlapped file output)
void ParallelHeatDistribution(float *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters &parameters,
                              string outputFileName);

/// Store time step into output file
void StoreDataIntoFile(hid_t h5fileId,
                       const float *data,
                       const size_t edgeSize,
                       const size_t snapshotId,
                       const size_t iteration);

/// Store time step into output file using parallel HDF5
void StoreDataIntoFileParallel(hid_t h5fileId,
                               const float *data,
                               const size_t edgeSize,
                               const size_t tileWidth, const size_t tileHeight,
                               const size_t tilePosX, const size_t tilePosY,
                               const size_t snapshotId,
                               const size_t iteration);


//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//

void ComputePoint(float *oldTemp,
                  float *newTemp,
                  float *params,
                  int *map,
                  size_t i,
                  size_t j,
                  size_t edgeSize,
                  float airFlowRate,
                  float coolerTemp) {
  // [i] Calculate neighbor indices
  const int center = i * edgeSize + j;
  const int top[2] = {center - (int) edgeSize, center - 2 * (int) edgeSize};
  const int bottom[2] = {center + (int) edgeSize, center + 2 * (int) edgeSize};
  const int left[2] = {center - 1, center - 2};
  const int right[2] = {center + 1, center + 2};

  // [ii] The reciprocal value of the sum of domain parameters for normalization
  const float frac = 1.0f / (params[top[0]] + params[top[1]] +
                             params[bottom[0]] + params[bottom[1]] +
                             params[left[0]] + params[left[1]] +
                             params[right[0]] + params[right[1]] +
                             params[center]);

  // [iii] Calculate new temperature in the grid point
  float pointTemp =
      oldTemp[top[0]] * params[top[0]] * frac +
      oldTemp[top[1]] * params[top[1]] * frac +
      oldTemp[bottom[0]] * params[bottom[0]] * frac +
      oldTemp[bottom[1]] * params[bottom[1]] * frac +
      oldTemp[left[0]] * params[left[0]] * frac +
      oldTemp[left[1]] * params[left[1]] * frac +
      oldTemp[right[0]] * params[right[0]] * frac +
      oldTemp[right[1]] * params[right[1]] * frac +
      oldTemp[center] * params[center] * frac;

  // [iv] Remove some of the heat due to air flow (5% of the new air)
  pointTemp = (map[center] == 0)
              ? (airFlowRate * coolerTemp) + ((1.0f - airFlowRate) * pointTemp)
              : pointTemp;

  newTemp[center] = pointTemp;
}

/**
 * Sequential version of the Heat distribution in heterogenous 2D medium
 * @param [out] seqResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 */
void SequentialHeatDistribution(float *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters &parameters,
                                string outputFileName) {
  // [1] Create a new output hdf5 file
  hid_t file_id = H5I_INVALID_HID;

  if (outputFileName != "") {
    if (outputFileName.find(".h5") == string::npos)
      outputFileName.append("_seq.h5");
    else
      outputFileName.insert(outputFileName.find_last_of("."), "_seq");

    file_id = H5Fcreate(outputFileName.c_str(),
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
    if (file_id < 0) ios::failure("Cannot create output file");
  }


  // [2] A temporary array is needed to prevent mixing of data form step t and t+1
  float *tempArray = (float *) _mm_malloc(materialProperties.nGridPoints *
                                          sizeof(float), DATA_ALIGNMENT);

  // [3] Init arrays
  for (size_t i = 0; i < materialProperties.nGridPoints; i++) {
    tempArray[i] = materialProperties.initTemp[i];
    seqResult[i] = materialProperties.initTemp[i];
  }

  // [4] t+1 values, t values
  float *newTemp = seqResult;
  float *oldTemp = tempArray;

  if (!parameters.batchMode)
    printf("Starting sequential simulation... \n");

  //---------------------- [5] start the stop watch ------------------------------//
  double elapsedTime = MPI_Wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;
  size_t printCounter = 1;

  // [6] Start the iterative simulation
  for (iteration = 0; iteration < parameters.nIterations; iteration++) {
    // [a] calculate one iteration of the heat distribution (skip the grid points at the edges)
    for (i = 2; i < materialProperties.edgeSize - 2; i++)
      for (j = 2; j < materialProperties.edgeSize - 2; j++)
        ComputePoint(oldTemp,
                     newTemp,
                     materialProperties.domainParams,
                     materialProperties.domainMap,
                     i, j,
                     materialProperties.edgeSize,
                     parameters.airFlowRate,
                     materialProperties.coolerTemp);

    // [b] Compute the average temperature in the middle column
    middleColAvgTemp = 0.0f;
    for (i = 0; i < materialProperties.edgeSize; i++)
      middleColAvgTemp += newTemp[i * materialProperties.edgeSize +
                                  materialProperties.edgeSize / 2];
    middleColAvgTemp /= materialProperties.edgeSize;

    // [c] Store time step in the output file if necessary
    if ((file_id != H5I_INVALID_HID) && ((iteration % parameters.diskWriteIntensity) == 0)) {
      StoreDataIntoFile(file_id,
                        newTemp,
                        materialProperties.edgeSize,
                        iteration / parameters.diskWriteIntensity,
                        iteration);
    }

    // [d] Swap new and old values
    swap(newTemp, oldTemp);

    // [e] Print progress and average temperature of the middle column
    if (((float) (iteration) >= (parameters.nIterations - 1) / 10.0f * (float) printCounter)
        && !parameters.batchMode) {
      printf("Progress %ld%% (Average Temperature %.2f degrees)\n",
             (iteration + 1) * 100L / (parameters.nIterations),
             middleColAvgTemp);
      ++printCounter;
    }
  } // for iteration

  //-------------------- stop the stop watch  --------------------------------//
  double totalTime = MPI_Wtime() - elapsedTime;

  // [7] Print final result
  if (!parameters.batchMode)
    printf("\nExecution time of sequential version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "seq",
           middleColAvgTemp, totalTime,
           totalTime / parameters.nIterations);

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // [8] Return correct results in the correct array
  if (iteration & 1)
    memcpy(seqResult, tempArray, materialProperties.nGridPoints * sizeof(float));

  _mm_free(tempArray);
} // end of SequentialHeatDistribution
//------------------------------------------------------------------------------

class CPUMatrices {
public:
    CPUMatrices(int cpus, size_t edge, int rank);
    ~CPUMatrices();
    void copyNewToOld();
    void scatter();
    void gather();
    void sendHaloBlocks();
    void recieveHaloBlocks();
    void printMyData();
    void waitHaloBlocks();
    void recieveHaloBlocksDomainParams();
    void sendHaloBlocksDomainParams();

    double elapsedTime;
    float *globalArray;
    float *globalDomainParams;
    int *globalDomainMap;

    float *localNewData;
    float *localData;
    int *localDomainMap;
    float *localDomainParams;


    int myRank;
    int myRow;
    int myCol;
    int myLeftNeighbour;
    int myRightNeighbour;
    int myTopNeighbour;
    int myBottomNeighbour;
    size_t edgeSize;
    size_t widthProcs;
    size_t heightProcs;
    size_t widthEdge;
    size_t heightEdge;
    vector<int> sendcounts;
    vector<int> displs;
    int frameSize[2];
    int frameSizeOfLocal[2];
    int tileSize[2];
    int tileSizeOfLocal[2];
    int starts[2];
    int startsOfLocal[2];

    int numOfCpus;

    MPI_Datatype typeFloat, subArrFloatType;
    MPI_Datatype localTypeFloat, localSubArrFloatType;
    MPI_Datatype typeInt, subArrIntType;
    MPI_Datatype localTypeInt, localSubArrIntType;
    MPI_Datatype localRows;
    MPI_Datatype localCols;

    MPI_Request requests[8];
    MPI_Status statuses[8];
    int requestsCounter = 0;

    MPI_Group old_group;
    MPI_Group new_group;
    MPI_Comm middleColComm;
    int middleColSize;
    int middleColMyRank;
    float middleColAverageRoot;
    int middleColOffset;

    int tagTopRows = 5;
    int tagBottomRows = 6;
    int tagLeftCols = 7;
    int tagRightCols = 8;

    int haloTopRowStart;
    int haloDataTopRowStart;
    int haloBottomRowStart;
    int haloDataBottomRowStar;

    int haloLeftColStart;
    int haloDataLeftColStart;
    int haloRightColStart;
    int haloDataRighColStart;
};

CPUMatrices::CPUMatrices(int cpus, size_t edge, int rank) {
  edgeSize = edge;
  myRank = rank;
  numOfCpus = cpus;

  if(numOfCpus == 1){
    widthProcs = 1;
    heightProcs = 1;
  } else {
    if(fmod(sqrt(numOfCpus),2.0) != 0.0) {
      widthProcs = (numOfCpus == 2) ? 1 : sqrt(numOfCpus/2);
      if (widthProcs % 2 != 0 && numOfCpus != 2) {
        throw runtime_error("Error wrong # of processes.");
      }
      heightProcs = widthProcs*2;
    } else {
      widthProcs = sqrt(numOfCpus);
      heightProcs = widthProcs;
    }
  }

  widthEdge = edgeSize / widthProcs;
  heightEdge = edgeSize / heightProcs;

  myCol = myRank % widthProcs;
  myRow = myRank / widthProcs;
  myLeftNeighbour = myRank-1;
  myRightNeighbour = myRank+1;
  myTopNeighbour = myRank - widthProcs;
  myBottomNeighbour = myRank + widthProcs;

  frameSize[0] = edgeSize;         /* global size */
  frameSize[1] = edgeSize;         /* global size */
  tileSize[0] = heightEdge;     /* local size o 1 vacsia na kazd000 */
  tileSize[1] = widthEdge;     /* local size o 1 vacsia na kazd000 */
  starts[0] = 0;
  starts[1] = 0;

  MPI_Type_create_subarray(2, frameSize, tileSize, starts, MPI_ORDER_C, MPI_FLOAT, &typeFloat);
  MPI_Type_create_resized(typeFloat, 0, (widthEdge)*sizeof(float), &subArrFloatType);
  MPI_Type_commit(&subArrFloatType);

  MPI_Type_create_subarray(2, frameSize, tileSize, starts, MPI_ORDER_C, MPI_INT, &typeInt);
  MPI_Type_create_resized(typeInt, 0, (widthEdge)*sizeof(int), &subArrIntType);
  MPI_Type_commit(&subArrIntType);


  frameSizeOfLocal[0] = heightEdge+4;         /* Tile size + 2 halo down + 2 halo up */
  frameSizeOfLocal[1] = widthEdge+4;         /* Tile size + 2 halo left + 2 halo right */
  tileSizeOfLocal[0] = heightEdge;     /* size of tile */
  tileSizeOfLocal[1] = widthEdge;     /* size of tile */
  startsOfLocal[0] = 2;
  startsOfLocal[1] = 2;

  localDomainMap = (int  *) _mm_malloc((frameSizeOfLocal[0])*(frameSizeOfLocal[1])*sizeof(int),   DATA_ALIGNMENT);
  localDomainParams = (float*) _mm_malloc((frameSizeOfLocal[0])*(frameSizeOfLocal[1])* sizeof(float), DATA_ALIGNMENT);
  localData = (float*) _mm_malloc((frameSizeOfLocal[0])*(frameSizeOfLocal[1])* sizeof(float), DATA_ALIGNMENT);
  localNewData = (float*) _mm_malloc((frameSizeOfLocal[0])*(frameSizeOfLocal[1])* sizeof(float), DATA_ALIGNMENT);
  std::fill(localDomainMap, localDomainMap + (frameSizeOfLocal[0])*(frameSizeOfLocal[1]), 0);
  std::fill(localDomainParams, localDomainParams + (frameSizeOfLocal[0])*(frameSizeOfLocal[1]), 0.0);
  std::fill(localData, localData + (frameSizeOfLocal[0])*(frameSizeOfLocal[1]), 0.0);
  std::fill(localNewData, localNewData + (frameSizeOfLocal[0])*(frameSizeOfLocal[1]), 0.0);

  MPI_Type_create_subarray(2, frameSizeOfLocal, tileSizeOfLocal, startsOfLocal, MPI_ORDER_C, MPI_FLOAT, &localTypeFloat);
  MPI_Type_create_resized(localTypeFloat, 0, (widthEdge)*sizeof(float), &localSubArrFloatType);
  MPI_Type_commit(&localSubArrFloatType);


  MPI_Type_create_subarray(2, frameSizeOfLocal, tileSizeOfLocal, startsOfLocal, MPI_ORDER_C, MPI_INT, &localSubArrIntType);
  MPI_Type_commit(&localSubArrIntType);

  MPI_Type_vector(2, widthEdge, frameSizeOfLocal[1], MPI_FLOAT, &localRows);
  MPI_Type_commit(&localRows);

  MPI_Type_vector(heightEdge, 2, frameSizeOfLocal[1], MPI_FLOAT, &localCols);
  MPI_Type_commit(&localCols);

  haloTopRowStart = 2;
  haloDataTopRowStart = 2*frameSizeOfLocal[1]+2;
  haloBottomRowStart = (2+heightEdge)*frameSizeOfLocal[1]+2;
  haloDataBottomRowStar = (heightEdge)*frameSizeOfLocal[1]+2;

  haloLeftColStart = 2*frameSizeOfLocal[1];
  haloDataLeftColStart = 2*frameSizeOfLocal[1]+2;
  haloRightColStart = 2*frameSizeOfLocal[1]+2+widthEdge;
  haloDataRighColStart = 2*frameSizeOfLocal[1]+widthEdge;

  //Average communicator
  vector<int> middleCols;
  middleCols.clear();
  middleCols.push_back(0);

  for (int i = 1; i < numOfCpus; ++i) {
    if(i % widthProcs == widthProcs/2)
      middleCols.push_back(i);
  }
  if ( (0 % widthProcs == widthProcs/2 && middleCols.size() != heightProcs ) ||
       (0 % widthProcs != widthProcs/2 && middleCols.size()-1 != heightProcs )) {
    printf("Error in creation of middle comm. Middle size = %zu height %zu\n", middleCols.size(),heightProcs);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  MPI_Comm_group(MPI_COMM_WORLD, &old_group);
  MPI_Group_incl(old_group, middleCols.size(), middleCols.data(), &new_group);
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &middleColComm);
  MPI_Comm_rank(MPI_COMM_WORLD, &middleColMyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &middleColSize);
  middleColAverageRoot = 0.0f;
  middleColOffset = (numOfCpus > 2) ? 2 : (2 + widthEdge/2);

  sendcounts.resize(numOfCpus);
  displs.resize(numOfCpus);
  if (myRank == 0) {

    for (int i=0; i<numOfCpus; i++)
      sendcounts[i] = 1;
    int disp = 0;
    for (int i=0; i<heightProcs; i++) {
      disp = heightEdge * widthProcs * i;
      for (int j=0; j<widthProcs; j++) {
        displs[i*widthProcs+j] = disp;
        disp += 1;
      }
    }
  }
}

void CPUMatrices::scatter() {
  MPI_Scatterv(globalArray, sendcounts.data(), displs.data(), subArrFloatType, &(localNewData[0]), 1, localSubArrFloatType, 0, MPI_COMM_WORLD);
  MPI_Scatterv(globalDomainParams, sendcounts.data(), displs.data(), subArrFloatType, &(localDomainParams[0]), 1, localSubArrFloatType, 0, MPI_COMM_WORLD);
  MPI_Scatterv(globalDomainMap, sendcounts.data(), displs.data(), subArrIntType, &(localDomainMap[0]), 1, localSubArrIntType, 0, MPI_COMM_WORLD);
}

void CPUMatrices::gather() {
  MPI_Gatherv(&(localNewData[0]), 1, localSubArrFloatType, globalArray, sendcounts.data(), displs.data(), subArrFloatType, 0, MPI_COMM_WORLD);
}

void CPUMatrices::sendHaloBlocks() {
  //send TOP rows
  if(myRow != 0) {
    MPI_Isend(&(localNewData[haloDataTopRowStart]), 1, localRows, myTopNeighbour, tagTopRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //send BOTTOM rows
  if (myRow != heightProcs-1) {
    MPI_Isend(&(localNewData[haloDataBottomRowStar]), 1, localRows, myBottomNeighbour, tagBottomRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
//  //Send LEFT col
  if (myCol != 0 ) {
    MPI_Isend(&(localNewData[haloDataLeftColStart]), 1, localCols, myLeftNeighbour, tagLeftCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //Send RIGHT col
  if (myCol != widthProcs-1 ) {
    MPI_Isend(&(localNewData[haloDataRighColStart]), 1, localCols, myRightNeighbour, tagRightCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
}

void CPUMatrices::recieveHaloBlocks() {
  //recieve BOTTOM rows
  if (myRow != heightProcs-1) {
    MPI_Irecv(&(localNewData[haloBottomRowStart]), 1, localRows, myBottomNeighbour, tagTopRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //recieve TOP rows
  if (myRow != 0) {
    MPI_Irecv(&(localNewData[haloTopRowStart]), 1, localRows, myTopNeighbour, tagBottomRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
//  //Recv RIGHT col
  if (myCol != widthProcs-1 ) {
    MPI_Irecv(&(localNewData[haloRightColStart]), 1, localCols, myRightNeighbour, tagLeftCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //Recv LEFT col
  if (myCol != 0 ) {
    MPI_Irecv(&(localNewData[haloLeftColStart]), 1, localCols, myLeftNeighbour, tagRightCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
}

void CPUMatrices::sendHaloBlocksDomainParams() {
  //send TOP rows
  if(myRow != 0) {
    MPI_Isend(&(localDomainParams[haloDataTopRowStart]), 1, localRows, myTopNeighbour, tagTopRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //send BOTTOM rows
  if (myRow != heightProcs-1) {
    MPI_Isend(&(localDomainParams[haloDataBottomRowStar]), 1, localRows, myBottomNeighbour, tagBottomRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
//  //Send LEFT col
  if (myCol != 0 ) {
    MPI_Isend(&(localDomainParams[haloDataLeftColStart]), 1, localCols, myLeftNeighbour, tagLeftCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //Send RIGHT col
  if (myCol != widthProcs-1 ) {
    MPI_Isend(&(localDomainParams[haloDataRighColStart]), 1, localCols, myRightNeighbour, tagRightCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
}

void CPUMatrices::recieveHaloBlocksDomainParams() {
  //recieve BOTTOM rows
  if (myRow != heightProcs-1) {
    MPI_Irecv(&(localDomainParams[haloBottomRowStart]), 1, localRows, myBottomNeighbour, tagTopRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //recieve TOP rows
  if (myRow != 0) {
    MPI_Irecv(&(localDomainParams[haloTopRowStart]), 1, localRows, myTopNeighbour, tagBottomRows, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
//  //Recv RIGHT col
  if (myCol != widthProcs-1 ) {
    MPI_Irecv(&(localDomainParams[haloRightColStart]), 1, localCols, myRightNeighbour, tagLeftCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
  //Recv LEFT col
  if (myCol != 0 ) {
    MPI_Irecv(&(localDomainParams[haloLeftColStart]), 1, localCols, myLeftNeighbour, tagRightCols, MPI_COMM_WORLD, &requests[requestsCounter++]);
  }
}

void CPUMatrices::waitHaloBlocks() {
  MPI_Waitall(requestsCounter, requests, statuses);
  requestsCounter = 0;
}

CPUMatrices::~CPUMatrices() {
  _mm_free(localDomainMap);
  _mm_free(localDomainParams);
  _mm_free(localData);
  MPI_Type_free(&subArrFloatType);
  MPI_Type_free(&subArrIntType);
}

void CPUMatrices::printMyData() {
  for (int p=0; p<numOfCpus; p++) {
    if (myRank == p) {
      printf("Local process on rank %d is:\n", myRank);
      for (int i=0; i< frameSizeOfLocal[0] * frameSizeOfLocal[1]; i++) {
        if(i % frameSizeOfLocal[1] == 0)
          printf("|");
        printf("%f ", localNewData[i]);
        if(i % frameSizeOfLocal[1] == frameSizeOfLocal[1]-1)
          printf("|\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void CPUMatrices::copyNewToOld() {
  memcpy(localData, localNewData, (frameSizeOfLocal[0])*(frameSizeOfLocal[1])*sizeof(float));
}


/**
 * Parallel version of the Heat distribution in heterogenous 2D medium
 * @param [out] parResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 * @note This is the function that students should implement.
 */
void ParallelHeatDistribution(float *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters &parameters,
                              string outputFileName) {
  // Get MPI rank and size
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  hid_t file_id = H5I_INVALID_HID;

  if (!parameters.useParallelIO) {
    // Serial I/O
    if (rank == 0 && outputFileName != "") {
      if (outputFileName.find(".h5") == string::npos)
        outputFileName.append("_par.h5");
      else
        outputFileName.insert(outputFileName.find_last_of("."), "_par");

      file_id = H5Fcreate(outputFileName.c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          H5P_DEFAULT);
      if (file_id < 0) ios::failure("Cannot create output file");
    }
  } else {
    // Parallel I/O
    if (outputFileName != "") {
      if (outputFileName.find(".h5") == string::npos)
        outputFileName.append("_par.h5");
      else
        outputFileName.insert(outputFileName.find_last_of("."), "_par");

      hid_t hPropList = H5Pcreate(H5P_FILE_ACCESS);
      H5Pset_fapl_mpio(hPropList, MPI_COMM_WORLD, MPI_INFO_NULL);

      file_id = H5Fcreate(outputFileName.c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          hPropList);
      H5Pclose(hPropList);
      if (file_id < 0) ios::failure("Cannot create output file");
    }
  }

  //--------------------------------------------------------------------------//
  //---------------- THE SECTION WHERE STUDENTS MAY ADD CODE -----------------//
  //--------------------------------------------------------------------------//
  size_t iteration;
  size_t i, j;
  float middleColAvgTemp = 0.0f;
  float middleColAverageRoot = 0.0f;
  size_t printCounter = 1;
  double elapsedTime;
  float *newTemp;
  float *oldTemp;

  CPUMatrices cpuMatrices(size, parameters.edgeSize, rank);
  cpuMatrices.globalArray = parResult;
  cpuMatrices.globalDomainMap = materialProperties.domainMap;
  cpuMatrices.globalDomainParams = materialProperties.domainParams;
  newTemp = cpuMatrices.localNewData;
  oldTemp = cpuMatrices.localData;

  int startPointCol = (cpuMatrices.myCol == 0) ? 4 : 2;
  int endPointCol = (cpuMatrices.myCol == cpuMatrices.widthProcs - 1) ? cpuMatrices.widthEdge : 2 +
                                                                                                cpuMatrices.widthEdge;
  int startPointRow = (cpuMatrices.myRow == 0) ? 4 : 2;
  int endPointRow = (cpuMatrices.myRow == cpuMatrices.heightProcs - 1) ? cpuMatrices.heightEdge : 2 +
                                                                                                  cpuMatrices.heightEdge;

#pragma omp parallel firstprivate(printCounter) private(iteration) private(middleColAverageRoot)
  {

    // [3] Init arrays
    if (rank == 0) {
#pragma omp for simd
      for (i = 0; i < materialProperties.nGridPoints; i++) {
        parResult[i] = materialProperties.initTemp[i];
      }
    }

#pragma omp master
    {
      cpuMatrices.scatter();
      cpuMatrices.sendHaloBlocks();
      cpuMatrices.recieveHaloBlocks();
      cpuMatrices.waitHaloBlocks();
      cpuMatrices.sendHaloBlocksDomainParams();
      cpuMatrices.recieveHaloBlocksDomainParams();
      cpuMatrices.waitHaloBlocks();
      cpuMatrices.copyNewToOld();

      if (!parameters.batchMode && rank == 0)
        printf("Starting parallel simulation... \n");

      //Meranie   //Ulozim cas pre vypocet dlzky behu
      if (rank == 0)
        elapsedTime = MPI_Wtime();
    }


    for (iteration = 0; iteration < parameters.nIterations; iteration++) {
#pragma omp master
      cpuMatrices.recieveHaloBlocks();
#pragma omp barrier
      //Calculate TOP rows
      if (cpuMatrices.myRow != 0) {
#pragma omp for firstprivate(oldTemp, newTemp) private(j)
        for (i = 2; i < 4; ++i) {
#pragma omp simd
          for (j = startPointCol; j < endPointCol; ++j) {
            ComputePoint(oldTemp, newTemp, cpuMatrices.localDomainParams,
                         cpuMatrices.localDomainMap, i, j, cpuMatrices.frameSizeOfLocal[1], parameters.airFlowRate,
                         materialProperties.coolerTemp);
          }
        }
      }
      //Calculate BOTTOM rows
      if (cpuMatrices.myRow != cpuMatrices.heightProcs - 1) {
#pragma omp for firstprivate(oldTemp, newTemp) private(j)
        for (i = cpuMatrices.heightEdge; i < cpuMatrices.heightEdge + 2; ++i) {
#pragma omp simd
          for (j = startPointCol; j < endPointCol; ++j) {
            ComputePoint(oldTemp, newTemp, cpuMatrices.localDomainParams,
                         cpuMatrices.localDomainMap, i, j, cpuMatrices.frameSizeOfLocal[1], parameters.airFlowRate,
                         materialProperties.coolerTemp);
          }
        }

      }
      //Calculate LEFT col
      if (cpuMatrices.myCol != 0) {
#pragma omp for firstprivate(oldTemp, newTemp) private(j)
        for (i = startPointRow; i < endPointRow; ++i) {
#pragma omp simd
          for (j = 2; j < 4; ++j) {
            ComputePoint(oldTemp, newTemp, cpuMatrices.localDomainParams,
                         cpuMatrices.localDomainMap, i, j, cpuMatrices.frameSizeOfLocal[1], parameters.airFlowRate,
                         materialProperties.coolerTemp);
          }
        }

      }
      //Calculate RIGHT col
      if (cpuMatrices.myCol != cpuMatrices.widthProcs - 1) {
#pragma omp for firstprivate(oldTemp, newTemp) private(j)
        for (i = startPointRow; i < endPointRow; ++i) {
#pragma omp simd
          for (j = cpuMatrices.widthEdge; j < 2 + cpuMatrices.widthEdge; ++j) {
            ComputePoint(oldTemp, newTemp, cpuMatrices.localDomainParams,
                         cpuMatrices.localDomainMap, i, j, cpuMatrices.frameSizeOfLocal[1], parameters.airFlowRate,
                         materialProperties.coolerTemp);
          }
        }

      }
#pragma omp master
      {
        cpuMatrices.sendHaloBlocks();
      }
#pragma omp barrier

#pragma omp for firstprivate(oldTemp, newTemp) private(j)
      for (i = 4; i < cpuMatrices.heightEdge; ++i) {
#pragma omp simd
        for (j = 4; j < cpuMatrices.widthEdge; ++j) {
          ComputePoint(oldTemp,
                       newTemp,
                       cpuMatrices.localDomainParams,
                       cpuMatrices.localDomainMap,
                       i, j,
                       cpuMatrices.frameSizeOfLocal[1],
                       parameters.airFlowRate,
                       materialProperties.coolerTemp);
        }
      }

#pragma omp master
      {
        cpuMatrices.waitHaloBlocks();
        middleColAvgTemp = 0.0f;

        if (rank == 0) {
          middleColAverageRoot = 0.0f;
        }
      }
#pragma omp barrier

      // [b] Compute the average temperature in the middle column
      if (cpuMatrices.myCol == cpuMatrices.widthProcs / 2 || rank == 0) {
        if (cpuMatrices.myCol == cpuMatrices.widthProcs / 2) {
#pragma omp for simd reduction(+:middleColAvgTemp)
          for (int i = 2; i < cpuMatrices.heightEdge + 2; i++)
            middleColAvgTemp += newTemp[i * cpuMatrices.frameSizeOfLocal[1] + cpuMatrices.middleColOffset];
        }
#pragma omp master
        MPI_Reduce(&middleColAvgTemp, &middleColAverageRoot, 1, MPI_FLOAT, MPI_SUM, 0, cpuMatrices.middleColComm);
      }

#pragma omp master
      {
        if (rank == 0) {
          middleColAverageRoot /= materialProperties.edgeSize;
          // [e] Print progress and average temperature of the middle column
          if (((float) (iteration) >= (parameters.nIterations - 1) / 10.0f * (float) printCounter)
              && !parameters.batchMode) {
            printf("Progress %ld%% (Average Temperature %.2f degrees)\n",
                   (iteration + 1) * 100L / (parameters.nIterations),
                   middleColAverageRoot);
            ++printCounter;
          }
        }



        //ZAPIS DO SUBORU
        // Doporuceny zpusob ukladani dat do vystupniho souboru
        // Tento kod by se vyskytoval v hlavni iteracni smycce (viz. sekvencni implementace)
        // Pro paralelni I/O neni treba prenaset data do 0. procesu (na rozdil od serioveho I/O).
        if (iteration % parameters.diskWriteIntensity == 0) {
          if (!parameters.useParallelIO) {
            // Serial I/O
            // store data to root
            // *** Zde posbirejte data do 0. procesu, ktery vytvarel vystupni soubor ***
            MPI_Barrier(MPI_COMM_WORLD);
            cpuMatrices.gather();
            // store time step in the output file if necessary
            if (rank == 0 && file_id != H5I_INVALID_HID) {
              StoreDataIntoFile(file_id,
                                parResult,
                                materialProperties.edgeSize,
                                iteration / parameters.diskWriteIntensity,
                                iteration);
            }
          } else {
            // Parallel I/O
            if (file_id != H5I_INVALID_HID) {
              StoreDataIntoFileParallel(file_id,
                                        newTemp,
                                        materialProperties.edgeSize,
                                        cpuMatrices.widthEdge + 4, cpuMatrices.heightEdge + 4,
                                        (cpuMatrices.widthEdge) * cpuMatrices.myCol,
                                        (cpuMatrices.heightEdge) * cpuMatrices.myRow,
                                        iteration / parameters.diskWriteIntensity, iteration);
            }
          }
        }

        swap(newTemp, oldTemp);
        //Need this because the cpuMatrices does have different pointers which are used to send data and recieve data
        swap(cpuMatrices.localNewData, cpuMatrices.localData);

      }//Koniec master pragma
    }

#pragma omp master
    {
      cpuMatrices.gather();

      //Meranie
      if (rank == 0) {
        double totalTime = MPI_Wtime() - elapsedTime;
        // [7] Print final result
        if (!parameters.batchMode)
          printf("\nExecution time of parallel version %.5f\n", totalTime);
        else
          printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "par",
                 middleColAverageRoot, totalTime,
                 totalTime / parameters.nIterations);
      }

    }

  }//Koniec pragma parallel


  // close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);
} // end of ParallelHeatDistribution
//------------------------------------------------------------------------------


/**
 * Store time step into output file (as a new dataset in Pixie format
 * @param [in] h5fileID   - handle to the output file
 * @param [in] data       - data to write
 * @param [in] edgeSize   - size of the domain
 * @param [in] snapshotId - snapshot id
 * @param [in] iteration  - id of iteration);
 */
void StoreDataIntoFile(hid_t h5fileId,
                       const float *data,
                       const size_t edgeSize,
                       const size_t snapshotId,
                       const size_t iteration) {
  hid_t dataset_id, dataspace_id, group_id, attribute_id;
  hsize_t dims[2] = {edgeSize, edgeSize};

  string groupName = "Timestep_" + to_string((unsigned long long) snapshotId);

  // Create a group named "/Timestep_snapshotId" in the file.
  group_id = H5Gcreate(h5fileId,
                       groupName.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


  // Create the data space. (2D matrix)
  dataspace_id = H5Screate_simple(2, dims, NULL);

  // create a dataset for temperature and write data
  string datasetName = "Temperature";
  dataset_id = H5Dcreate(group_id,
                         datasetName.c_str(),
                         H5T_NATIVE_FLOAT,
                         dataspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id,
           H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           data);

  // close dataset
  H5Sclose(dataspace_id);


  // write attribute
  string atributeName = "Time";
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2(group_id, atributeName.c_str(),
                            H5T_IEEE_F64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT);

  double snapshotTime = double(iteration);
  H5Awrite(attribute_id, H5T_IEEE_F64LE, &snapshotTime);
  H5Aclose(attribute_id);


  // Close the dataspace.
  H5Sclose(dataspace_id);

  // Close to the dataset.
  H5Dclose(dataset_id);

  // Close the group.
  H5Gclose(group_id);
} // end of StoreDataIntoFile
//------------------------------------------------------------------------------

/**
 * Store time step into output file using parallel version of HDF5
 * @param [in] h5fileId   - handle to the output file
 * @param [in] data       - data to write
 * @param [in] edgeSize   - size of the domain
 * @param [in] tileWidth  - width of the tile
 * @param [in] tileHeight - height of the tile
 * @param [in] tilePosX   - position of the tile in the grid (X-dir)
 * @param [in] tilePosY   - position of the tile in the grid (Y-dir)
 * @param [in] snapshotId - snapshot id
 * @param [in] iteration  - id of iteration
 */
void StoreDataIntoFileParallel(hid_t h5fileId,
                               const float *data,
                               const size_t edgeSize,
                               const size_t tileWidth, const size_t tileHeight,
                               const size_t tilePosX, const size_t tilePosY,
                               const size_t snapshotId,
                               const size_t iteration) {
  hid_t dataset_id, dataspace_id, group_id, attribute_id, memspace_id;
  const hsize_t dims[2] = {edgeSize, edgeSize};
  const hsize_t offset[2] = {tilePosY, tilePosX};
  const hsize_t tile_dims[2] = {tileHeight, tileWidth};
  const hsize_t core_dims[2] = {tileHeight - 4, tileWidth - 4};
  const hsize_t core_offset[2] = {2, 2};

  string groupName = "Timestep_" + to_string((unsigned long) snapshotId);

  // Create a group named "/Timestep_snapshotId" in the file.
  group_id = H5Gcreate(h5fileId,
                       groupName.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Create the data space in the output file. (2D matrix)
  dataspace_id = H5Screate_simple(2, dims, NULL);

  // create a dataset for temperature and write data
  string datasetName = "Temperature";
  dataset_id = H5Dcreate(group_id,
                         datasetName.c_str(),
                         H5T_NATIVE_FLOAT,
                         dataspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // create the data space in memory representing local tile. (2D matrix)
  memspace_id = H5Screate_simple(2, tile_dims, NULL);

  // select appropriate block of the local tile. (without halo zones)
  H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, core_offset, NULL, core_dims, NULL);

  // select appropriate block of the output file, where local tile will be placed.
  H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, core_dims, NULL);

  // setup collective write using MPI parallel I/O
  hid_t hPropList = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(hPropList, H5FD_MPIO_COLLECTIVE);

  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, hPropList, data);

  // close memory spaces and property list
  H5Sclose(memspace_id);
  H5Sclose(dataspace_id);
  H5Pclose(hPropList);

  // write attribute
  string attributeName = "Time";
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2(group_id, attributeName.c_str(),
                            H5T_IEEE_F64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT);

  double snapshotTime = double(iteration);
  H5Awrite(attribute_id, H5T_IEEE_F64LE, &snapshotTime);
  H5Aclose(attribute_id);

  // close the dataspace
  H5Sclose(dataspace_id);

  // close the dataset and the group
  H5Dclose(dataset_id);
  H5Gclose(group_id);
}
//------------------------------------------------------------------------------

/**
 * Main function of the project
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char *argv[]) {
  int rank, size;

  ParseCommandline(argc, argv, parameters);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get MPI rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, true);
    parameters.edgeSize = materialProperties.edgeSize;

    parameters.PrintParameters();
  } else {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, false);
    parameters.edgeSize = materialProperties.edgeSize;
  }

  if (parameters.edgeSize % size) {
    if (rank == 0)
      printf("ERROR: number of MPI processes is not a divisor of N\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (parameters.IsRunSequntial()) {
    if (rank == 0) {
      // Memory allocation for output matrices.
      seqResult = (float *) _mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);

      SequentialHeatDistribution(seqResult,
                                 materialProperties,
                                 parameters,
                                 parameters.outputFileName);
    }
  }

  if (parameters.IsRunParallel()) {
    // Memory allocation for output matrix.
    if (rank == 0)
      parResult = (float *) _mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);
    else
      parResult = NULL;

    ParallelHeatDistribution(parResult,
                             materialProperties,
                             parameters,
                             parameters.outputFileName);
  }

  // Validate the outputs
  if (parameters.IsValidation() && rank == 0) {
    if (parameters.debugFlag) {
      printf("---------------- Sequential results ---------------\n");
      PrintArray(seqResult, materialProperties.edgeSize);

      printf("----------------- Parallel results ----------------\n");
      PrintArray(parResult, materialProperties.edgeSize);
    }

    if (VerifyResults(seqResult, parResult, parameters, 0.001f))
      printf("Verification OK\n");
    else
      printf("Verification FAILED\n");
  }

  /* Memory deallocation*/
  _mm_free(seqResult);
  _mm_free(parResult);

  MPI_Finalize();

  return EXIT_SUCCESS;
} // end of main
//------------------------------------------------------------------------------