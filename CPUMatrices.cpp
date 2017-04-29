//
// Created by archie on 4/24/17.
//

#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <algorithm>
#include "CPUMatrices.h"
#include "BasicRoutines.h"
#include <immintrin.h>

using namespace std;

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
//        cout << "i " <<i<<" j "<<j<<" displs" <<displs[i*widthProcs+j]<<endl;
        disp += 1;
      }
    }
  }
}

void CPUMatrices::scatter() {
  MPI_Scatterv(globalArray, sendcounts.data(), displs.data(), subArrFloatType, &(localNewData[0]), 1, localSubArrFloatType, 0, MPI_COMM_WORLD);
  MPI_Scatterv(globalDomainParams, sendcounts.data(), displs.data(), subArrFloatType, &(localDomainParams[0]), 1, localSubArrFloatType, 0, MPI_COMM_WORLD);
  MPI_Scatterv(globalDomainMap, sendcounts.data(), displs.data(), subArrIntType, &(localDomainMap[0]), 1, localSubArrIntType, 0, MPI_COMM_WORLD);
//  printMyData();
}

void CPUMatrices::gather() {
  /* it all goes back to process 0 */
//  vector<MPI_Status> status(numOfCpus+1);
//  vector<MPI_Request> request(numOfCpus+1);
//  int counterRequests = 0;
//  vector<int> displs2(numOfCpus);

  MPI_Gatherv(&(localNewData[0]), 1, localSubArrFloatType, globalArray, sendcounts.data(), displs.data(), subArrFloatType, 0, MPI_COMM_WORLD);

  //MPI_Gatherv did deadlock on the Salomon. Also Irecv and Isend did deadlock whe sent all data to 0 rank
  //Following is workaround send from 0 to 0 with Isend and Irecv and then use Send and Recv from other ranks
//  if (myRank == 0) {
//    int disp = 0;
//    for (int i=0; i<heightProcs; i++) {
//
//      disp = heightEdge * edgeSize * i;
//
//      for (int j=0; j<widthProcs; j++) {
//
//        displs2[i*widthProcs+j] = disp;
////        cout << "i " <<i<<" j "<<j<<" displs " <<displs2[i*widthProcs+j]<<endl;
//        disp += 1* widthEdge;
//      }
//    }
//  }
//
//  if(myRank == 0) {
//    MPI_Isend(&(localNewData[0]), 1, localSubArrFloatType, 0, 1, MPI_COMM_WORLD, &request[counterRequests++]);
////    cout << myRank << " " << counterRequests << endl;
////    cout << "reccving from " << 0 << " to " << displs2[0] << endl;
//    MPI_Irecv(&(globalArray[displs2[0]]), 1, subArrFloatType, 0, 1, MPI_COMM_WORLD, &request[counterRequests++]);
////    cout << "requests " << counterRequests << endl;
//    MPI_Waitall(counterRequests, request.data(), status.data());
//  }
//
//  if (myRank != 0) {
//    MPI_Send(&(localNewData[0]), 1, localSubArrFloatType, 0, 1, MPI_COMM_WORLD);
////    cout << myRank << " sent request" << endl;
//  }
//
//  if(myRank == 0){
//    for (int i = 1; i < numOfCpus; ++i) {
////      cout << "reccving from " << i << " to " << displs2[i] << endl;
//      MPI_Recv(&(globalArray[displs2[i]]), 1, subArrFloatType, i, 1, MPI_COMM_WORLD, &(status[0]));
//    }
//  }
//
////  if (myRank == 0) {
////    printf("Processed grid:\n");
////    for (int i=0; i<edgeSize*edgeSize; i++) {
////      cout << globalArray[i] << " ";
////      if(i % edgeSize == edgeSize-1)
////        cout << endl;
////    }
////
////  }
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



