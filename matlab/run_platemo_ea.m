function [archiveX,archiveF,archiveC,archiveFE,lower,upper] = run_platemo_ea( ...
    platemoRoot,algorithmName,problemName,N,M,D,maxFE,seed,problemParameters)
%RUN_PLATEMO_EA Run a generic PlatEMO EA with an unbounded evaluation log.

    validateattributes(N,{'numeric'},{'scalar','integer','positive'});
    validateattributes(M,{'numeric'},{'scalar','integer','>=',2});
    validateattributes(maxFE,{'numeric'},{'scalar','integer','>=',N});
    assert(isfolder(platemoRoot),'PlatEMO root does not exist: %s',platemoRoot);
    addpath(genpath(platemoRoot));
    addpath(fileparts(mfilename('fullpath')));
    assert(~isempty(which('platemo')),'platemo.m was not found under the supplied root.');
    assert(~isempty(which(algorithmName)),'Unknown PlatEMO algorithm: %s',algorithmName);
    assert(~isempty(which(problemName)),'Unknown PlatEMO problem: %s',problemName);

    global EA_MODEL_ARCHIVE
    EA_MODEL_ARCHIVE = struct();
    problemSpec = {@EA_ModelRecordedProblem,problemName,problemParameters,seed};
    args = {'algorithm',str2func(algorithmName),'problem',problemSpec, ...
        'N',N,'M',M,'maxFE',maxFE,'save',0};
    if D > 0
        args = [args,{'D',D}]; %#ok<AGROW>
    end
    platemo(args{:});

    assert(isfield(EA_MODEL_ARCHIVE,'X'),'Recorder did not receive any evaluations.');
    assert(size(EA_MODEL_ARCHIVE.X,1) == size(EA_MODEL_ARCHIVE.F,1), ...
        'Recorded decisions and objectives are not aligned.');
    assert(numel(EA_MODEL_ARCHIVE.FE) == size(EA_MODEL_ARCHIVE.X,1), ...
        'Recorded FE indices are not aligned.');
    archiveX = EA_MODEL_ARCHIVE.X;
    archiveF = EA_MODEL_ARCHIVE.F;
    archiveC = EA_MODEL_ARCHIVE.C;
    archiveFE = EA_MODEL_ARCHIVE.FE;
    lower = EA_MODEL_ARCHIVE.lower;
    upper = EA_MODEL_ARCHIVE.upper;
end

