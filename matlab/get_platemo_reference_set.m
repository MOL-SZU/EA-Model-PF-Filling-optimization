function Z = get_platemo_reference_set( ...
    platemoRoot,problemName,N,M,D,problemParameters,referencePointNum)
%GET_PLATEMO_REFERENCE_SET Query the benchmark's dense true PF samples.
%   This is an oracle operation intended for benchmark validation. It is
%   not generally available for real-world problems with an unknown PF.
    addpath(genpath(platemoRoot));
    Problem = create_platemo_problem(problemName,N,M,D,problemParameters);
    Z = Problem.GetOptimum(referencePointNum);
    assert(~isempty(Z),'Problem.GetOptimum returned an empty reference set.');
    assert(size(Z,2) == Problem.M,'PF reference set has the wrong dimension.');
    assert(all(isfinite(Z(:))),'PF reference set contains NaN or Inf.');
end
