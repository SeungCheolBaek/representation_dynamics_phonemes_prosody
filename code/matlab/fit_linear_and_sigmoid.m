%% Fitting linear and sigmoid functions to behavior response
%
% written by S-C. Baek
% update: 12.12.2024
%
%
%% Path setting

% where is this file?
MATDIR = fileparts(mfilename('fullpath'));

% make sure that we are in the directory where this .m file is located
cd(MATDIR)

% path to main
cd('../..')
MAINDIR = pwd();

% go back MATDIR
cd(MATDIR)

% path to data
DATADIR = fullfile(MAINDIR, 'data');


%% Configuration

% subjects
s = dir([DATADIR, '/sub-*']);
n_subjects = length(s);
subjects = cell(n_subjects,1);
for i = 1:n_subjects 
    subjects{i} = s(i).name;
end
clear s

% tasks
tasks = {'phoneme'; 'prosody'};
n_tasks = length(tasks);


%% fit liear and sigmoid functions to individual behavioral responses

% levels
x = 1:5;
n_levels = length(x);

% loop over subjects
for subi = 1:n_subjects

    % loop over linguistic units
    for ti = 1:n_tasks

        % load behavioral data
        fpath = fullfile(DATADIR, subjects{subi}, 'behavior');
        load(fullfile(fpath, ['task_', tasks{ti}, '.mat']))

        % accuracy averaged across voices
        y = mean([behav(1).acc; behav(2).acc], 1);

        % prepare data into a right form
        [xData, yData] = prepareCurveData( x, y );


        % ---------- linear fitting ---------- %
        % Set up fittype and options.
        ft = fittype('poly1');
        opts = fitoptions('poly1');

        % Fit model to data.
        [fitresult, gof] = fit( xData, yData, ft, opts );

        % fitted parameters
        params = coeffvalues(fitresult); % y=ax+b: slope (a) and intercept (b)

        % store fitting information
        linfit.intercept = params(2);
        linfit.slope     = params(1);
        linfit.rsq       = gof.rsquare;
        linfit.adjrsq    = gof.adjrsquare;
        clear ft opts fitresult gof params


        % ---------- sigmoid fitting ---------- %
        % Set up fittype and options.
        ft = fittype( '1/(1+exp((a-x)*b))', 'independent', 'x', 'dependent', 'y' );
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.StartPoint = [quantile(xData,0.5), 1]; 

        % Fit model to data.
        [fitresult, gof] = fit( xData, yData, ft, opts );

        % fitted parameters
        params = coeffvalues(fitresult);

        % store fitting information
        sigfit.x50    = params(1); % 50%-point threshold (a)
        sigfit.slope  = params(2); % slope (b)
        sigfit.rsq    = gof.rsquare;
        sigfit.adjrsq = gof.adjrsquare;
        clear ft opts fitresult gof params


        % data structure to save the fitting information
        fitinfo.x = x;
        fitinfo.y = y;
        fitinfo.linfit = linfit;
        fitinfo.sigfit = sigfit;

        % save fitted information
        fname = fullfile(fpath, ['task_fit_', tasks{ti}, '.mat']);
        if ~exist(fname)
            disp([' saving ' fname ' ... '])
            save(fname, 'fitinfo')
        else
            disp([fname ' - already exists'])
        end

    end % end task-loop

end % end subject-loop