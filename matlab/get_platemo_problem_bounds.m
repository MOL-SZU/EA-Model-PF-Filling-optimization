function [lower,upper] = get_platemo_problem_bounds( ...
    platemoRoot,problemName,N,M,D,problemParameters)
%GET_PLATEMO_PROBLEM_BOUNDS Query decision bounds without consuming FE.
    addpath(genpath(platemoRoot));
    Problem = create_platemo_problem(problemName,N,M,D,problemParameters);
    lower = double(Problem.lower(:)');
    upper = double(Problem.upper(:)');
    assert(numel(lower) == Problem.D && numel(upper) == Problem.D, ...
        'Problem returned decision bounds with the wrong dimension.');
    assert(all(isfinite(lower)) && all(isfinite(upper)) && all(upper > lower), ...
        'Problem returned invalid decision bounds.');
end
