#include "get_prolong.h"

void get_prolong(
	const Eigen::MatrixXd & VO,
	const Eigen::MatrixXi & FO,
  const int & tarF,
	const int & dec_type,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double> & P)
{
  using namespace Eigen;
	using namespace std;

  // decimate 
	VectorXi IM, FIM;
	vector<single_collapse_data> decInfo;
	vector<vector<int>> decIM;
	VectorXi J; 
	SSP_decimate(VO,FO,tarF, dec_type, V,F,J, IM, decInfo, decIM, FIM);

  // get indices to query 
	MatrixXd BC(VO.rows(),3); BC.setZero();
	MatrixXi BF(VO.rows(),3); BF.setZero();
	VectorXi FIdx(VO.rows()); FIdx.setZero();
	// get all the find point barycentric (looks like [1,0,0])
	for (int fIdx=0; fIdx<FO.rows(); fIdx++)
	{
		for (int ii = 0; ii<FO.cols(); ii++)
		{
			int vIdx = FO(fIdx,ii);
			if (BC.row(vIdx).sum() == 0.0)
			{
				BC(vIdx,ii) = 1;
				BF.row(vIdx) = FO.row(fIdx);
				FIdx(vIdx) = fIdx;	
			}
		}
	}

  // query fine vertices to the coarse
  query_fine_to_coarse(decInfo, IM, decIM, FIM, BC, BF, FIdx);

  // assemble P
  vector<Triplet<double>> IJV;
  IJV.reserve(BC.rows() * BC.cols());

  for (int c = 0; c < BC.cols(); c++)
  {
    for (int r =0; r < BC.rows(); r++)
    {
      IJV.push_back(Triplet<double>(r, BF(r,c), BC(r,c)));
    }
  }
  P.resize(VO.rows(), V.rows());
  P.setFromTriplets(IJV.begin(), IJV.end());
}

void get_prolong_and_restriction(
	const Eigen::MatrixXd & VO,
	const Eigen::MatrixXi & FO,
  const int & tarF,
	const int & dec_type,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double> & P,
  Eigen::SparseMatrix<double> & S)
{
	using namespace Eigen;
	using namespace std;

  // decimate 
	VectorXi IM, FIM;
	vector<single_collapse_data> decInfo;
	vector<vector<int>> decIM;
	VectorXi J; 
	SSP_decimate(VO,FO,tarF, dec_type, V,F,J, IM, decInfo, decIM, FIM);

  // get indices to query 
	MatrixXd BC_P(VO.rows(),3); BC_P.setZero();
	MatrixXi BF_P(VO.rows(),3); BF_P.setZero();
	VectorXi FIdx_P(VO.rows()); FIdx_P.setZero();
	// get all the find point barycentric (looks like [1,0,0])
	for (int fIdx=0; fIdx<FO.rows(); fIdx++)
	{
		for (int ii = 0; ii<FO.cols(); ii++)
		{
			int vIdx = FO(fIdx,ii);
			if (BC_P.row(vIdx).sum() == 0.0)
			{
				BC_P(vIdx,ii) = 1;
				BF_P.row(vIdx) = FO.row(fIdx);
				FIdx_P(vIdx) = fIdx;	
			}
		}
	}

  // query fine vertices to the coarse
  query_fine_to_coarse(decInfo, IM, decIM, FIM, BC_P, BF_P, FIdx_P);

  // assemble P
  vector<Triplet<double>> IJV_P;
  IJV_P.reserve(BC_P.rows() * BC_P.cols());

  for (int c = 0; c < BC_P.cols(); c++)
  {
    for (int r =0; r < BC_P.rows(); r++)
    {
      IJV_P.push_back(Triplet<double>(r, BF_P(r,c), BC_P(r,c)));
    }
  }
  P.resize(VO.rows(), V.rows());
  P.setFromTriplets(IJV_P.begin(), IJV_P.end());

	// get barycentric coordinates on the coarse mesh for querying (I set it to the coarse vertices in this case)
	MatrixXd BC_S(V.rows(),3); BC_S.setZero();
	MatrixXi BF_S(V.rows(),3); BF_S.setZero();
	VectorXi FIdx_S(V.rows()); FIdx_S.setZero();
	for (int fIdx=0; fIdx<F.rows(); fIdx++)
	{
		for (int ii = 0; ii<F.cols(); ii++)
		{
			int vIdx = F(fIdx,ii);
			if (BC_S.row(vIdx).sum() == 0.0)
			{
				BC_S(vIdx,ii) = 1;
				BF_S.row(vIdx) = F.row(fIdx);
				FIdx_S(vIdx) = fIdx;	
			}
		}
	}

	// query coarse to fine 
	query_coarse_to_fine(decInfo, IM, decIM, J, BC_S, BF_S, FIdx_S);

	// assemble S
  vector<Triplet<double>> IJV_S;
  IJV_S.reserve(BC_S.rows() * BC_S.cols());

  for (int c = 0; c < BC_S.cols(); c++)
  {
    for (int r =0; r < BC_S.rows(); r++)
    {
      IJV_S.push_back(Triplet<double>(r, BF_S(r,c), BC_S(r,c)));
    }
  }
  S.resize(V.rows(), VO.rows());
  S.setFromTriplets(IJV_S.begin(), IJV_S.end());

}

void get_prolong_block(
	const Eigen::MatrixXd & VO,
	const Eigen::MatrixXi & FO,
  const int & tarF,
	const int & dec_type,
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double> & P)
{
  using namespace Eigen;
	using namespace std;

  // decimate 
	VectorXi IM, FIM;
	vector<single_collapse_data> decInfo;
	vector<vector<int>> decIM;
	VectorXi J; 
	SSP_decimate(VO,FO,tarF, dec_type, V,F,J, IM, decInfo, decIM, FIM);

  // get indices to query 
	MatrixXd BC(VO.rows(),3); BC.setZero();
	MatrixXi BF(VO.rows(),3); BF.setZero();
	VectorXi FIdx(VO.rows()); FIdx.setZero();
	// get all the find point barycentric (looks like [1,0,0])
	for (int fIdx=0; fIdx<FO.rows(); fIdx++)
	{
		for (int ii = 0; ii<FO.cols(); ii++)
		{
			int vIdx = FO(fIdx,ii);
			if (BC.row(vIdx).sum() == 0.0)
			{
				BC(vIdx,ii) = 1;
				BF.row(vIdx) = FO.row(fIdx);
				FIdx(vIdx) = fIdx;	
			}
		}
	}

  // query fine vertices to the coarse
  query_fine_to_coarse(decInfo, IM, decIM, FIM, BC, BF, FIdx);

	// assemble P
  vector<Triplet<double>> IJV;
  IJV.reserve(3 * BC.rows() * BC.cols());

  for (int c = 0; c < BC.cols(); c++)
  {
    for (int r =0; r < BC.rows(); r++)
    {
      IJV.push_back(Triplet<double>(3*r  , 3*BF(r,c)  , BC(r,c)));
			IJV.push_back(Triplet<double>(3*r+1, 3*BF(r,c)+1, BC(r,c)));
			IJV.push_back(Triplet<double>(3*r+2, 3*BF(r,c)+2, BC(r,c)));
    }
  }
  P.resize(3*VO.rows(), 3*V.rows());
  P.setFromTriplets(IJV.begin(), IJV.end());
}