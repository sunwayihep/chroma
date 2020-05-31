
// -*- C++ -*-
/*! \file
 *  \QUDA MULTIGRID MdagM Wilson solver.
 */

#ifndef __syssolver_mdagm_quda_multigrid_wilson_h__
#define __syssolver_mdagm_quda_multigrid_wilson_h__

#include "chroma_config.h"

#ifdef BUILD_QUDA
#include <quda.h>

#include "handle.h"
#include "state.h"
#include "syssolver.h"
#include "linearop.h"
#include "actions/ferm/fermbcs/simple_fermbc.h"
#include "actions/ferm/fermstates/periodic_fermstate.h"
#include "actions/ferm/invert/quda_solvers/syssolver_quda_multigrid_wilson_params.h"
#include "actions/ferm/linop/eoprec_wilson_linop_w.h"
#include "meas/gfix/temporal_gauge.h"
#include "io/aniso_io.h"
#include <string>
#include "actions/ferm/invert/quda_solvers/quda_mg_utils.h"
#include "actions/ferm/invert/mg_solver_exception.h"

#include "util/gauge/reunit.h"
#ifdef QDP_IS_QDPJIT
#include "actions/ferm/invert/quda_solvers/qdpjit_memory_wrapper.h"
#endif
#include "update/molecdyn/predictor/zero_guess_predictor.h"
#include "update/molecdyn/predictor/quda_predictor.h"
#include "meas/inline/io/named_objmap.h"
#include "lmdagm.h"

//#include <util_quda.h>

namespace Chroma
{

	//! Richardson system solver namespace
	namespace MdagMSysSolverQUDAMULTIGRIDWilsonEnv
	{
		//! Register the syssolver
		bool registerAll();
	}



	//! Solve a Wilson Fermion System using the QUDA inverter
	/*! \ingroup invert
	 *** WARNING THIS SOLVER WORKS FOR Wilson FERMIONS ONLY ***
	 */

	class MdagMSysSolverQUDAMULTIGRIDWilson : public MdagMSystemSolver<LatticeFermion>
	{
		public:
			typedef LatticeFermion T;
			typedef LatticeColorMatrix U;
			typedef multi1d<LatticeColorMatrix> Q;

			typedef LatticeFermionF TF;
			typedef LatticeColorMatrixF UF;
			typedef multi1d<LatticeColorMatrixF> QF;

			typedef LatticeFermionF TD;
			typedef LatticeColorMatrixF UD;
			typedef multi1d<LatticeColorMatrixF> QD;

			typedef WordType<T>::Type_t REALT;
			//! Constructor
			/*!
			 * \param M_        Linear operator ( Read )
			 * \param invParam  inverter parameters ( Read )
			 */
			MdagMSysSolverQUDAMULTIGRIDWilson(Handle< LinearOperator<T> > A_,
					Handle< FermState<T,Q,Q> > state_,
					const SysSolverQUDAMULTIGRIDWilsonParams& invParam_) :
				A(A_), invParam(invParam_)
		{
			StopWatch init_swatch;
			init_swatch.reset();
			init_swatch.start();

			std::ostringstream solver_string_stream;
			solver_string_stream << "QUDA_MULTIGRID_WILSON_MDAGM_SOLVER( Mass = " << invParam.WilsonParams.Mass << ", Id = "
				<< invParam.SaveSubspaceID << " ):";
			solver_string = solver_string_stream.str();

			QDPIO::cout << solver_string << "Initializing" << std::endl;

			// FOLLOWING INITIALIZATION in test QUDA program

			// 1) work out cpu_prec, cuda_prec, cuda_prec_sloppy
			int s = sizeof( WordType<T>::Type_t );
			if (s == 4) {
				cpu_prec = QUDA_SINGLE_PRECISION;
			}
			else {
				cpu_prec = QUDA_DOUBLE_PRECISION;
			}


			// Work out GPU precision
			switch( invParam.cudaPrecision ) {
				case HALF:
					gpu_prec = QUDA_HALF_PRECISION;
					break;
				case SINGLE:
					gpu_prec = QUDA_SINGLE_PRECISION;
					break;
				case DOUBLE:
					gpu_prec = QUDA_DOUBLE_PRECISION;
					break;
				default:
					gpu_prec = cpu_prec;
					break;
			}

			// Work out GPU Sloppy precision
			// Default: No Sloppy
			switch( invParam.cudaSloppyPrecision ) {
				case HALF:
					gpu_half_prec = QUDA_HALF_PRECISION;
					break;
				case SINGLE:
					gpu_half_prec = QUDA_SINGLE_PRECISION;
					break;
				case DOUBLE:
					gpu_half_prec = QUDA_DOUBLE_PRECISION;
					break;
				default:
					gpu_half_prec = gpu_prec;
					break;
			}

			// 2) pull 'new; GAUGE and Invert params
			q_gauge_param = newQudaGaugeParam();
			quda_inv_param = newQudaInvertParam();

			// 3) set lattice size
			const multi1d<int>& latdims = Layout::subgridLattSize();

			q_gauge_param.X[0] = latdims[0];
			q_gauge_param.X[1] = latdims[1];
			q_gauge_param.X[2] = latdims[2];
			q_gauge_param.X[3] = latdims[3];

			// 4) - deferred (anisotropy)

			// 5) - set QUDA_WILSON_LINKS, QUDA_GAUGE_ORDER
			q_gauge_param.type = QUDA_WILSON_LINKS;
#ifndef BUILD_QUDA_DEVIFACE_GAUGE
			q_gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER; // gauge[mu], p
#else
			QDPIO::cout << "MDAGM Using QDP-JIT gauge order" << std::endl;
			q_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION;
			q_gauge_param.gauge_order = QUDA_QDPJIT_GAUGE_ORDER;
#endif

			// 6) - set t_boundary
			// Convention: BC has to be applied already
			// This flag just tells QUDA that this is so,
			// so that QUDA can take care in the reconstruct
			if( invParam.AntiPeriodicT ) {
				q_gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
			}
			else {
				q_gauge_param.t_boundary = QUDA_PERIODIC_T;
			}

			// Set cpu_prec, cuda_prec, reconstruct and sloppy versions
			q_gauge_param.cpu_prec = cpu_prec;
			q_gauge_param.cuda_prec = gpu_prec;


			switch( invParam.cudaReconstruct ) {
				case RECONS_NONE:
					q_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
					break;
				case RECONS_8:
					q_gauge_param.reconstruct = QUDA_RECONSTRUCT_8;
					break;
				case RECONS_12:
					q_gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
					break;
				default:
					q_gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
					break;
			};

			q_gauge_param.cuda_prec_sloppy = gpu_half_prec;

			switch( invParam.cudaSloppyReconstruct ) {
				case RECONS_NONE:
					q_gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
					break;
				case RECONS_8:
					q_gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_8;
					break;
				case RECONS_12:
					q_gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
					break;
				default:
					q_gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
					break;
			};


			// Gauge fixing:

			// These are the links
			// They may be smeared and the BC's may be applied
			Q links_single(Nd);

			// Now downcast to single prec fields.
			for(int mu=0; mu < Nd; mu++) {
				links_single[mu] = (state_->getLinks())[mu];
			}

			// GaugeFix
			if( invParam.axialGaugeP ) {
				QDPIO::cout << "Fixing Temporal Gauge" << std::endl;
				temporalGauge(links_single, GFixMat, Nd-1);
				for(int mu=0; mu < Nd; mu++){
					links_single[mu] = GFixMat*(state_->getLinks())[mu]*adj(shift(GFixMat, FORWARD, mu));
				}
				q_gauge_param.gauge_fix = QUDA_GAUGE_FIXED_YES;
			}
			else {
				// No GaugeFix
				q_gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;  // No Gfix yet
			}

			// deferred 4) Gauge Anisotropy
			const AnisoParam_t& aniso = invParam.WilsonParams.anisoParam;
			if( aniso.anisoP ) {                     // Anisotropic case
				Real gamma_f = aniso.xi_0 / aniso.nu;
				q_gauge_param.anisotropy = toDouble(gamma_f);
			}
			else {
				q_gauge_param.anisotropy = 1.0;
			}

			// MAKE FSTATE BEFORE RESCALING links_single
			// Because the clover term expects the unrescaled links...
			Handle<FermState<T,Q,Q> > fstate( new PeriodicFermState<T,Q,Q>(links_single));

			if( aniso.anisoP ) {                     // Anisotropic case
				multi1d<Real> cf=makeFermCoeffs(aniso);
				for(int mu=0; mu < Nd; mu++) {
					links_single[mu] *= cf[mu];
				}
			}

			// Now onto the inv param:
			// Dslash type

			quda_inv_param.dslash_type = QUDA_WILSON_DSLASH;

			// Invert type:
			switch( invParam.solverType ) {
				case CG:
					quda_inv_param.inv_type = QUDA_CG_INVERTER;
					break;
				case BICGSTAB:
					quda_inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
					break;
				case GCR:
					quda_inv_param.inv_type = QUDA_GCR_INVERTER;
					break;
				default:
					QDPIO::cerr << "Unknown Solver type" << std::endl;
					QDP_abort(1);
					break;
			}


			Real diag_mass;
			{
				// auto is C++11 so I don't have to remember all the silly typenames
				auto wlparams = invParam.WilsonParams;

				auto aniso = wlparams.anisoParam;

				Real ff = where(aniso.anisoP, aniso.nu / aniso.xi_0, Real(1));
				diag_mass = 1 + (Nd-1)*ff + wlparams.Mass;
			}


			quda_inv_param.kappa = static_cast<double>(1)/(static_cast<double>(2)*toDouble(diag_mass));

			quda_inv_param.tol = toDouble(invParam.RsdTarget);
			quda_inv_param.maxiter = invParam.MaxIter;
			quda_inv_param.reliable_delta = toDouble(invParam.Delta);
			quda_inv_param.pipeline = invParam.Pipeline;

			// Solution type
			//quda_inv_param.solution_type = QUDA_MATPC_SOLUTION;
			//Taken from invert test.
			quda_inv_param.solution_type = QUDA_MATPC_SOLUTION;
			quda_inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;

			quda_inv_param.matpc_type = QUDA_MATPC_ODD_ODD;

			quda_inv_param.dagger = QUDA_DAG_NO;
			quda_inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

			quda_inv_param.cpu_prec = cpu_prec;
			quda_inv_param.cuda_prec = gpu_prec;
			quda_inv_param.cuda_prec_sloppy = gpu_half_prec;

			quda_inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
			quda_inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

#ifndef BUILD_QUDA_DEVIFACE_SPINOR
			quda_inv_param.dirac_order = QUDA_DIRAC_ORDER;
#else
			QDPIO::cout << "MDAGM Using QDP-JIT spinor order" << std::endl;
			quda_inv_param.dirac_order    = QUDA_QDPJIT_DIRAC_ORDER;
			quda_inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;
			quda_inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;
#endif
			// Autotuning
			if( invParam.tuneDslashP ) {
				QDPIO::cout << "Enabling Dslash Autotuning" << std::endl;

				quda_inv_param.tune = QUDA_TUNE_YES;
			}
			else {
				QDPIO::cout << "Disabling Dslash Autotuning" << std::endl;

				quda_inv_param.tune = QUDA_TUNE_NO;
			}


			// Setup padding
			multi1d<int> face_size(4);
			face_size[0] = latdims[1]*latdims[2]*latdims[3]/2;
			face_size[1] = latdims[0]*latdims[2]*latdims[3]/2;
			face_size[2] = latdims[0]*latdims[1]*latdims[3]/2;
			face_size[3] = latdims[0]*latdims[1]*latdims[2]/2;

			int max_face = face_size[0];
			for(int i=1; i <=3; i++) {
				if ( face_size[i] > max_face ) {
					max_face = face_size[i];
				}
			}


			q_gauge_param.ga_pad = max_face;
			// PADDING
			quda_inv_param.sp_pad = 0;
			quda_inv_param.cl_pad = 0;

			if( !invParam.MULTIGRIDParamsP )  {
				QDPIO::cout << solver_string << "ERROR: MG Solver had MULTIGRIDParamsP set to false" << std::endl;
				QDP_abort(1);
			}
			if( invParam.MULTIGRIDParamsP ) {
				QDPIO::cout << "Setting MULTIGRID solver params" << std::endl;
				// Dereference handle
				const MULTIGRIDSolverParams& ip = *(invParam.MULTIGRIDParams);

				// Set preconditioner precision
				switch( ip.prec ) {
					case HALF:
						quda_inv_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
						quda_inv_param.clover_cuda_prec_precondition = QUDA_HALF_PRECISION;
						q_gauge_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
						break;

					case SINGLE:
						quda_inv_param.cuda_prec_precondition = QUDA_SINGLE_PRECISION;
						quda_inv_param.clover_cuda_prec_precondition = QUDA_SINGLE_PRECISION;
						q_gauge_param.cuda_prec_precondition = QUDA_SINGLE_PRECISION;
						break;

					case DOUBLE:
						quda_inv_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
						quda_inv_param.clover_cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
						q_gauge_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
						break;
					default:
						quda_inv_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
						quda_inv_param.clover_cuda_prec_precondition = QUDA_HALF_PRECISION;
						q_gauge_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
						break;
				}

				switch( ip.reconstruct ) {
					case RECONS_NONE:
						q_gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
						break;
					case RECONS_8:
						q_gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_8;
						break;
					case RECONS_12:
						q_gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_12;
						break;
					default:
						q_gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_12;
						break;
				};
			}

			// Set up the links
			void* gauge[4];

#ifndef BUILD_QUDA_DEVIFACE_GAUGE
			for(int mu=0; mu < Nd; mu++) {
				gauge[mu] = (void *)&(links_single[mu].elem(all.start()).elem().elem(0,0).real());
			}
#else
			GetMemoryPtrGauge(gauge,links_single);
			//gauge[mu] = GetMemoryPtr( links_single[mu].getId() );
			//QDPIO::cout << "MDAGM CUDA gauge[" << mu << "] in = " << gauge[mu] << "\n";
#endif

			loadGaugeQuda((void *)gauge, &q_gauge_param);

			MULTIGRIDSolverParams ip = *(invParam.MULTIGRIDParams);
			quda_inv_param.tol_precondition = toDouble(ip.tol[0]);
			quda_inv_param.maxiter_precondition = ip.maxIterations[0];
			quda_inv_param.gcrNkrylov = ip.outer_gcr_nkrylov;

			//Replacing above with what's in the invert test.
			switch( ip.schwarzType ) {
				case ADDITIVE_SCHWARZ :
					quda_inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
					break;
				case MULTIPLICATIVE_SCHWARZ :
					quda_inv_param.schwarz_type = QUDA_MULTIPLICATIVE_SCHWARZ;
					break;
				default:
					quda_inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
					break;
			}
			quda_inv_param.precondition_cycle = 1;
			//Invert test always sets this to 1.

			if( invParam.verboseP ) {
				quda_inv_param.verbosity = QUDA_VERBOSE;
				quda_inv_param.verbosity_precondition = QUDA_VERBOSE;
			}
			else {
				quda_inv_param.verbosity = QUDA_SUMMARIZE;
				quda_inv_param.verbosity_precondition = QUDA_SILENT;
			}
			//MG is the only option.
			quda_inv_param.inv_type_precondition = QUDA_MG_INVERTER;
			//New invert test changes here.
			quda_inv_param.Ls = 1;

			// Check whether there exists a MG subspace already
			if(TheNamedObjMap::Instance().check(invParam.SaveSubspaceID))
			{
				StopWatch update_swatch;
				update_swatch.reset();
				update_swatch.start();
				// Subspace ID exists add it to mg_state
				QDPIO::cout << solver_string << "Recovering subspace..." << std::endl;
				subspace_pointers = TheNamedObjMap::Instance().getData<QUDAMGUtils::MGSubspacePointers* >(invParam.SaveSubspaceID);
				for(int j=0; j<ip.mg_levels-1;++j){
					(subspace_pointers->mg_param).setup_maxiter_refresh[j] = 0;
				}
				updateMultigridQuda(subspace_pointers->preconditioner, &(subspace_pointers->mg_param));

				update_swatch.stop();
				QDPIO::cout << solver_string << "subspace_update_time = "
					<<update_swatch.getTimeInSeconds() << " sec. " << std::endl;
			}else{
				// Subspace not exist, creating
				StopWatch create_swatch;
				create_swatch.reset();
				create_swatch.start();
				QDPIO::cout << solver_string << "Creating Subspace" << std::endl;

				// setup the multigrid solver
				subspace_pointers = QUDAMGUtils::create_subspace<T>(invParam);
				XMLBufferWriter file_xml;
				push(file_xml, "FileXML");
				pop(file_xml);

				int foo = 5;
				XMLBufferWriter record_xml;
				push(record_xml, "RecordXML");
				write(record_xml, "foo", foo);
				pop(record_xml);

				TheNamedObjMap::Instance().create<QUDAMGUtils::MGSubspacePointers* >(invParam.SaveSubspaceID);
				TheNamedObjMap::Instance().get(invParam.SaveSubspaceID).setFileXML(file_xml);
				TheNamedObjMap::Instance().get(invParam.SaveSubspaceID).setRecordXML(record_xml);
				TheNamedObjMap::Instance().getData<QUDAMGUtils::MGSubspacePointers* >(invParam.SaveSubspaceID) = subspace_pointers;

				create_swatch.stop();
				QDPIO::cout << solver_string << " subspace_create_time = "
					<< create_swatch.getTimeInSeconds() << " sec. " << std::endl;
			}
			quda_inv_param.preconditioner = subspace_pointers->preconditioner;

			init_swatch.stop();
			QDPIO::cout << solver_string << " init_time = "
				<< init_swatch.getTimeInSeconds() << " sec. " << std::endl;
		}


			//! Destructor is automatic
			~MdagMSysSolverQUDAMULTIGRIDWilson()
			{
				QDPIO::cout << "Destructing" << std::endl;
				quda_inv_param.preconditioner = nullptr;
				subspace_pointers = nullptr;
				freeGaugeQuda();
			}

			//! Return the subset on which the operator acts
			const Subset& subset() const {return A->subset();}

			//! Solver the linear system
			/*!
			 * \param psi      solution ( Modify )
			 * \param chi      source ( Read )
			 * \return syssolver results
			 */
			SystemSolverResults_t operator()(T& psi, const T& chi) const
			{
				SystemSolverResults_t res;
				SystemSolverResults_t res1;
				SystemSolverResults_t res2;

				// MdagM used to check the QUDA solution
				Handle<LinearOperator<T> > MdagM(new MdagMLinOp<T>(A));

				START_CODE();
				StopWatch swatch;
				swatch.start();

				psi = zero;
				QDPIO::cout << std::endl;
				QDPIO::cout << solver_string << " Two Step Solve" << std::endl;
				QDPIO::cout << solver_string << " Solve Y" << std::endl;

				// it seems the full site order is not implemented in Quda for QDP-JIT
				// so we can just solve the even-odd preconditioned system using Quda Multigrid
				// if we need the solution of full system, then first solve on the odd site and
				// then reconstruct the full system
				T g5chi = zero;
				g5chi = Gamma(Nd * Nd - 1) * chi;
				T Y_prime = zero;
				T Y = zero;

				Double norm2chi;
				bool solution_good = true;
				
				// downcast to EvenOddPrecWilsonLinOp
				EvenOddPrecWilsonLinOp* eoA = dynamic_cast<EvenOddPrecWilsonLinOp*>(&(*A));
				if(invParam.isMatPC){
					QDPIO::cout << solver_string << " Solving the EO Preconditioned MdagM system" << std::endl;

					res1 = qudaInvert(g5chi, Y_prime);
					Y = Gamma(Nd * Nd - 1) * Y_prime;
					// Check solution Y
					{

						norm2chi = sqrt(norm2(chi, A->subset()));

						T r;
						r[A->subset()] = chi;
						T tmp;
						(*A)(tmp, Y, MINUS);
						r[A->subset()] -= tmp;

						res1.resid = sqrt(norm2(r, A->subset()));
						QDPIO::cout << solver_string << " Y-solve: ||r||=" << res1.resid <<" ||r||/||b||="
							<< res1.resid/norm2chi << std::endl;

						if(toBool(res1.resid/norm2chi > invParam.RsdToleranceFactor * invParam.RsdTarget)){
							QDPIO::cout << solver_string << " Y Solve Failed, currently just ABORT" << std::endl;

							solution_good = false;

							MGSolverException convergence_fail(invParam.WilsonParams.Mass,
									invParam.SaveSubspaceID,
									res1.n_count,
									Real(res1.resid/norm2chi),
									invParam.RsdTarget * invParam.RsdToleranceFactor);
							throw convergence_fail;
						}
					}
				}else{
					QDPIO::cout << solver_string << " Solving the Unpreconditioned MdagM system" << std::endl;
					// prepare the source for even-odd preconditioned system
					QDPIO::cout << solver_string << " Prepare the source for even-odd preconditioned Quda Multigrid" << std::endl;
					T chi_tmp;
					{
						T tmp1, tmp2;
						eoA->evenEvenInvLinOp(tmp1, g5chi, PLUS);
						eoA->oddEvenLinOp(tmp2, tmp1, PLUS);
						chi_tmp[rb[1]] = g5chi - tmp2;
					}
					// do the inversion on odd site
					res1 = qudaInvert(chi_tmp, Y_prime);
					// reconstruct the full solution 
					QDPIO::cout << solver_string << " Reconstruction the solution from even-odd preconditioned Quda Multigrid" << std::endl;
					{
						T tmp1, tmp2;
						eoA->evenOddLinOp(tmp1, Y_prime, PLUS);
						tmp2[rb[0]] = g5chi - tmp1;
						eoA->evenEvenInvLinOp(Y_prime, tmp2, PLUS);
					}
					Y = Gamma(Nd * Nd - 1) * Y_prime;

					// Check solution Y
					{
						norm2chi = sqrt(norm2(chi));

						T r;
						eoA->unprecLinOp(r, Y, MINUS);
						r -= chi;

						res1.resid = sqrt(norm2(r));
						QDPIO::cout << solver_string << " Y-solve: ||r||=" << res1.resid <<" ||r||/||b||="
							<< res1.resid/norm2chi << std::endl;

						if(toBool(res1.resid/norm2chi > invParam.RsdToleranceFactor * invParam.RsdTarget)){
							QDPIO::cout << solver_string << " Y Solve Failed, currently just ABORT" << std::endl;

							solution_good = false;

							MGSolverException convergence_fail(invParam.WilsonParams.Mass,
									invParam.SaveSubspaceID,
									res1.n_count,
									Real(res1.resid/norm2chi),
									invParam.RsdTarget * invParam.RsdToleranceFactor);
							throw convergence_fail;
						}
					}

				}
				
				QDPIO::cout << solver_string << " Solve X" <<std::endl;

				if(invParam.isMatPC){
					res2 = qudaInvert(Y, psi);
					// Check solution X
					{
						norm2chi = sqrt(norm2(chi, A->subset()));

						T r;
						r[A->subset()] = chi;
						T tmp;
						(*MdagM)(tmp, psi, PLUS);
						r[A->subset()] -= tmp;

						res2.resid = sqrt(norm2(r, A->subset()));
						QDPIO::cout << solver_string << " X-solve: ||r||=" << res2.resid <<" ||r||/||b||="
							<< res2.resid/norm2chi << std::endl;

						if(toBool(res2.resid/norm2chi > invParam.RsdToleranceFactor * invParam.RsdTarget)){
							QDPIO::cout << solver_string << " X Solve Failed, currently just ABORT" << std::endl;

							solution_good = false;

							MGSolverException convergence_fail(invParam.WilsonParams.Mass,
									invParam.SaveSubspaceID,
									res2.n_count,
									Real(res2.resid/norm2chi),
									invParam.RsdTarget * invParam.RsdToleranceFactor);
							throw convergence_fail;
						}
					}
				}else{
					QDPIO::cout << solver_string << " Prepare the source for even-odd preconditioned Quda Multigrid" << std::endl;
					// prepare the source for even-odd preconditioned system
					T chi_tmp;
					{
						T tmp1, tmp2;
						eoA->evenEvenInvLinOp(tmp1, Y, PLUS);
						eoA->oddEvenLinOp(tmp2, tmp1, PLUS);
						chi_tmp[rb[1]] = Y - tmp2;
					}
					// do the inversion on odd site
					res2 = qudaInvert(chi_tmp, psi);
					// reconstruct the full solution 
					QDPIO::cout << solver_string << " Reconstruction the solution from even-odd preconditioned Quda Multigrid" << std::endl;
					{
						T tmp1, tmp2;
						eoA->evenOddLinOp(tmp1, psi, PLUS);
						tmp2[rb[0]] = Y - tmp1;
						eoA->evenEvenInvLinOp(psi, tmp2, PLUS);
					}

					// Check solution X
					{
						norm2chi = sqrt(norm2(chi));

						T tmp, r;
						eoA->unprecLinOp(tmp, psi, PLUS);
						eoA->unprecLinOp(r, tmp, MINUS);
						r -= chi;

						res2.resid = sqrt(norm2(r));
						QDPIO::cout << solver_string << " X-solve: ||r||=" << res2.resid <<" ||r||/||b||="
							<< res2.resid/norm2chi << std::endl;

						if(toBool(res2.resid/norm2chi > invParam.RsdToleranceFactor * invParam.RsdTarget)){
							QDPIO::cout << solver_string << " X Solve Failed, currently just ABORT" << std::endl;

							solution_good = false;

							MGSolverException convergence_fail(invParam.WilsonParams.Mass,
									invParam.SaveSubspaceID,
									res2.n_count,
									Real(res2.resid/norm2chi),
									invParam.RsdTarget * invParam.RsdToleranceFactor);
							throw convergence_fail;
						}
					}
				}

				swatch.stop();
				double time = swatch.getTimeInSeconds();

				res.n_count = res1.n_count + res2.n_count;
				res.resid = res2.resid;
				Double rel_resid = res.resid / norm2chi;


				QDPIO::cout << solver_string  << " total iterations: " << res1.n_count << " + "
					<< res2.n_count << " = " << res.n_count
					<< " Rsd = " << res.resid << " Relative Rsd = " << rel_resid 
					<< " Total time: " << time << "(s)" << std::endl;
				QDPIO::cout << std::endl;

				END_CODE();
				return res;
			}

			SystemSolverResults_t operator()(T& psi, const T& chi, Chroma::AbsChronologicalPredictor4D<T>& predictor) const
			{
				SystemSolverResults_t res;
				START_CODE();

				this->operator()(psi, chi);

				END_CODE();

				return res;
			}


		private:
			// Hide default constructor
			MdagMSysSolverQUDAMULTIGRIDWilson() {}

#if 1
			Q links_orig;
#endif

			U GFixMat;
			QudaPrecision_s cpu_prec;
			QudaPrecision_s gpu_prec;
			QudaPrecision_s gpu_half_prec;

			Handle< LinearOperator<T> > A;
			mutable SysSolverQUDAMULTIGRIDWilsonParams invParam;
			QudaGaugeParam q_gauge_param;
			mutable QudaInvertParam quda_inv_param;
			mutable QudaInvertParam mg_inv_param;
			QudaMultigridParam mg_param;
			mutable QUDAMGUtils::MGSubspacePointers* subspace_pointers;
			SystemSolverResults_t qudaInvert( const T& chi_s, T& psi_s) const ;
			std::string solver_string;
	};


} // End namespace

#endif // BUILD_QUDA
#endif 

