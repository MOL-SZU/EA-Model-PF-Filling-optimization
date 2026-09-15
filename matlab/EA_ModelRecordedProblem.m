classdef EA_ModelRecordedProblem < PROBLEM
%EA_MODELRECORDEDPROBLEM Generic decorator recording every true evaluation.
%   The original problem class, its parameter vector, and seed are supplied
%   through PlatEMO's problem parameter cell. No optimizer is modified.

    properties(Access = private)
        InnerProblem
        Seed
    end

    methods
        function obj = EA_ModelRecordedProblem(varargin)
            obj@PROBLEM(varargin{:});
        end

        function Setting(obj)
            global EA_MODEL_ARCHIVE
            assert(numel(obj.parameter) >= 3, ...
                'Recorder requires problem name, problem parameters, and seed.');
            problemName = obj.parameter{1};
            problemParameters = obj.parameter{2};
            obj.Seed = double(obj.parameter{3});
            if isa(problemName,'function_handle')
                problemHandle = problemName;
            else
                problemHandle = str2func(char(problemName));
            end

            args = {'N',obj.N,'M',obj.M,'maxFE',obj.maxFE};
            if ~isempty(obj.D) && obj.D > 0
                args = [args,{'D',obj.D}]; %#ok<AGROW>
            end
            if ~isempty(problemParameters)
                parameterCell = num2cell(double(problemParameters(:)'));
                args(end+1:end+2) = {'parameter',parameterCell};
            end
            obj.InnerProblem = problemHandle(args{:});
            obj.M        = obj.InnerProblem.M;
            obj.D        = obj.InnerProblem.D;
            obj.encoding = obj.InnerProblem.encoding;
            obj.lower    = obj.InnerProblem.lower;
            obj.upper    = obj.InnerProblem.upper;

            EA_MODEL_ARCHIVE.X = zeros(0,obj.D);
            EA_MODEL_ARCHIVE.F = zeros(0,obj.M);
            EA_MODEL_ARCHIVE.C = zeros(0,0);
            EA_MODEL_ARCHIVE.FE = zeros(0,1);
            EA_MODEL_ARCHIVE.lower = obj.lower;
            EA_MODEL_ARCHIVE.upper = obj.upper;
            rng(obj.Seed,'twister');
        end

        function Population = Evaluation(obj,varargin)
            global EA_MODEL_ARCHIVE
            PopDec = obj.InnerProblem.CalDec(varargin{1});
            PopObj = obj.InnerProblem.CalObj(PopDec);
            PopCon = obj.InnerProblem.CalCon(PopDec);
            assert(size(PopDec,1) == size(PopObj,1), 'Decision/objective row mismatch.');
            assert(all(isfinite(PopDec(:))) && all(isfinite(PopObj(:))), ...
                'A true evaluation returned NaN or Inf.');
            n = size(PopDec,1);
            firstFE = obj.FE + 1;
            lastFE = obj.FE + n;
            EA_MODEL_ARCHIVE.X = [EA_MODEL_ARCHIVE.X;PopDec];
            EA_MODEL_ARCHIVE.F = [EA_MODEL_ARCHIVE.F;PopObj];
            if isempty(EA_MODEL_ARCHIVE.C)
                EA_MODEL_ARCHIVE.C = PopCon;
            else
                EA_MODEL_ARCHIVE.C = [EA_MODEL_ARCHIVE.C;PopCon];
            end
            EA_MODEL_ARCHIVE.FE = [EA_MODEL_ARCHIVE.FE;(firstFE:lastFE)'];
            Population = SOLUTION(PopDec,PopObj,PopCon,varargin{2:end});
            obj.FE = lastFE;
        end

        function PopDec = CalDec(obj,PopDec)
            PopDec = obj.InnerProblem.CalDec(PopDec);
        end

        function PopObj = CalObj(obj,PopDec)
            PopObj = obj.InnerProblem.CalObj(PopDec);
        end

        function PopCon = CalCon(obj,PopDec)
            PopCon = obj.InnerProblem.CalCon(PopDec);
        end

        function R = GetOptimum(obj,N)
            R = obj.InnerProblem.GetOptimum(N);
        end

        function R = GetPF(obj)
            R = obj.InnerProblem.GetPF();
        end
    end
end
