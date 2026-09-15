function [repairedX,F,C] = evaluate_platemo_problem( ...
    platemoRoot,problemName,N,M,D,problemParameters,X)
%EVALUATE_PLATEMO_PROBLEM Repair and truly evaluate candidates in PlatEMO.
    addpath(genpath(platemoRoot));
    Problem = create_platemo_problem(problemName,N,M,D,problemParameters);
    repairedX = Problem.CalDec(double(X));
    F = Problem.CalObj(repairedX);
    C = Problem.CalCon(repairedX);
    assert(size(repairedX,1) == size(F,1),'Decision/objective row mismatch.');
    assert(all(isfinite(repairedX(:))) && all(isfinite(F(:))), ...
        'True evaluation returned NaN or Inf.');
end
