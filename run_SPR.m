function X_warp = run_SPR(X,Y)
% This function run the SPR algorithm to fit the two point cloud
% X is the original point cloud and Y is the target for fitting
    SPR_opt.method = 'nonrigid';   %'nonrigid','nonrigid_lowrank'
    SPR_opt.viz = 1;   % disable visualization to speed up!
    SPR_opt.max_it = 150; SPR_opt.tol = -1;  % disable tolerance check, only max_it --> same iterations
    SPR_opt.outliers = 0;
    SPR_opt.knn = 20;
    SPR_opt.tau = 500;
    SPR_opt.beta = 2;
    SRP_opt.lambda = 3;
    SPR_opt.tau_annealing_handle = @(iter, max_it)  0.97^iter; 
    SPR_opt.lambda_annealing_handle = @(iter, max_it) 0.97^iter;
    [SPR_Transform, ~] = SPR_register(Y, X, SPR_opt); % CPD warp Y to X, fliped!
    X_warp = SPR_Transform.Y;
end