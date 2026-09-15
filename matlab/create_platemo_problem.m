function Problem = create_platemo_problem(problemName,N,M,D,problemParameters)
%CREATE_PLATEMO_PROBLEM Instantiate an original PlatEMO problem consistently.
    problemHandle = str2func(problemName);
    args = {'N',N,'M',M,'maxFE',realmax};
    if D > 0
        args = [args,{'D',D}]; %#ok<AGROW>
    end
    if ~isempty(problemParameters)
        parameterCell = num2cell(double(problemParameters(:)'));
        args(end+1:end+2) = {'parameter',parameterCell};
    end
    Problem = problemHandle(args{:});
end
