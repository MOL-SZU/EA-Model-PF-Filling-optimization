function [X,F] = run_mmeawi_mmf1(N,maxFE,platemoRoot)
%RUN_MMEAWI_MMF1 Legacy example using the generic archive recorder.
    if nargin < 3 || isempty(platemoRoot)
        platemoRoot = getenv('PLATEMO_PATH');
    end
    assert(~isempty(platemoRoot), ...
        'Pass platemoRoot or define the PLATEMO_PATH environment variable.');
    addpath(fullfile(fileparts(mfilename('fullpath')),'matlab'));
    [archiveX,archiveF] = run_platemo_ea(platemoRoot,'MMEAWI','MMF1', ...
        double(N),2,0,double(maxFE),1,[]);
    frontNo = NDSort(archiveF,1);
    X = archiveX(frontNo == 1,:);
    F = archiveF(frontNo == 1,:);
end
