clc;
clear;
close all;

format long;
%% Load data
data = readtable('data/2014-01-02_2018-12-31.csv');

% Assuming data columns represent different stocks, with dates in the first column
stockPrices = data{:, 2:end};  % Skip the first column which is the date

% Compute daily returns
dailyReturns = diff(stockPrices) ./ stockPrices(1:end-1, :);

%% Remove Outlier
% Calculate mean and standard deviation of the mean returns
mean_returns = mean(dailyReturns)';
mu = mean(mean_returns);
sigma = std(mean_returns);

% Compute Z-score and threshold for identifying outliers
z_scores = (mean_returns - mu) / sigma;
z_threshold = 5;

% Identify non-outlier indices and remove outliers from mean returns
non_outlier_indices = abs(z_scores) <= z_threshold;
mean_returns = mean_returns(non_outlier_indices);

% Exclude stocks corresponding to the outliers
dailyReturns = dailyReturns(:, non_outlier_indices);
cov_matrix = cov(dailyReturns);

% Calculate statistics after removing outliers
max_returns = max(mean_returns);
min_returns = min(mean_returns);
mean_returns_val = mean(mean_returns);
max_variance = max(diag(cov_matrix));
min_variance = min(diag(cov_matrix));
mean_variance = mean(diag(cov_matrix));

[numDays, numStocks] = size(dailyReturns);

%% Efficient Frontier
% Generate efficient frontier using quadratic programming
num_portfolios = 500; % Number of portfolios on the efficient frontier
target_returns = linspace(min(mean_returns), max(mean_returns), num_portfolios);
efficient_frontier_risks = zeros(num_portfolios, 1);
efficient_frontier_returns = zeros(num_portfolios, 1);
weights_matrix = zeros(num_portfolios, numStocks); % Initialize the weights matrix

% Define options for quadprog
options = optimoptions('quadprog', 'Display', 'off');

for i = 1:num_portfolios
    % Define the quadratic and linear terms of the objective function
    H = cov_matrix;
    f = [];

    % Define equality and inequality constraints
    Aeq = [mean_returns'; ones(1, numStocks)];
    beq = [target_returns(i); 1];
    A = [];
    b = [];
    lb = zeros(numStocks, 1);
    ub = ones(numStocks, 1);

    % Solve the quadratic programming problem
    [weights, risk] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

    % Store the return and risk
    efficient_frontier_returns(i) = weights' * mean_returns;
    efficient_frontier_risks(i) = weights' * cov_matrix * weights;
    
    % Store the weights
    weights_matrix(i, :) = weights';
end

% Combine returns, risks, and weights into one matrix
results = [efficient_frontier_returns, efficient_frontier_risks, weights_matrix];

% save csv
writematrix(results, 'data/efficient_frontier_weights.csv');

%% PSO parameters
% Initialize parameters
num_particles = 100;
num_iterations = 5000;
num_assets = numStocks;
num_selected_assets = 10; %round(numStocks*0.1); % Cardinality constraint
epsilon = (1/num_selected_assets)/5;
delta = (1/num_selected_assets)*5;

% Inertia and acceleration parameters
w_max = 0.9;
w_min = 0.4;
c1_max = 2.5;
c1_min = 0.5;
c2_max = 0.5;
c2_min = 2.5;

% Mutation parameters
mutation_probability = 0.1;
b = 5;

% Boundary constraints
lower_bounds = zeros(1, num_assets);
upper_bounds = ones(1, num_assets);

%% PSO
%Functions
% Risk-return trade-off parameters
risk_aversion_values = linspace(0, 1, 50); % Varying lambda values

% delta function 
delta_fun = @(t, x, num_iterations, b) x * (1 - rand * (1 - t / num_iterations)^b);

% PSO loop for each lambda value
PSO_portfolios = zeros(numel(risk_aversion_values), 3 + num_assets);

for lambda = 1:numel(risk_aversion_values)
    % num_restart = 0;

    risk_aversion = risk_aversion_values(lambda); % Update risk aversion for current iteration
    
    % Initialize particle positions and velocities within boundary constraints
    particle_positions = rand(num_particles, num_assets);
    particle_positions = lower_bounds + particle_positions .* (upper_bounds - lower_bounds);
    particle_velocities = zeros(num_particles, num_assets);
    
    % Binary selection variables
    binary_positions = zeros(num_particles, num_assets);
    
    for i = 1:num_particles
        selected_indices = randperm(num_assets, num_selected_assets);
        binary_positions(i, selected_indices) = 1;
    end
    
    personal_best_positions = particle_positions;
    personal_best_scores = inf(num_particles, 1);
    global_best_position = zeros(1, num_assets);
    global_best_score = inf;

    % Initialize termination criterion
    num_iterations_without_improvement = 0;
    prev_best_score = inf;

    % PSO loop
    for iter = 1:num_iterations
        % fprintf('Starting iteration %d for lambda = %.2f\n', iter, risk_aversion_values(lambda));

        % Time-variant inertia weight
        w = (w_min - w_max) * (num_iterations - iter) / num_iterations + w_max; 
        
        % Time-variant acceleration coefficients
        c1 = c1_max - (c1_max - c1_min) * (iter / num_iterations);
        c2 = c2_min + (c2_max - c2_min) * (iter / num_iterations);
    
        for i = 1:num_particles
            % Update velocities
            particle_velocities(i, :) = w * particle_velocities(i, :) + ...
                c1 * rand * (personal_best_positions(i, :) - particle_positions(i, :)) + ...
                c2 * rand * (global_best_position - particle_positions(i, :));
            
            % Update positions
            if iter < num_iterations/3
                % Apply reflection strategy to prevent leaving search space domain
                particle_positions(i, :) = particle_positions(i, :) + particle_velocities(i, :);
                lower_out_of_bounds = particle_positions(i, :) < lower_bounds;
                upper_out_of_bounds = particle_positions(i, :) > upper_bounds;
                particle_positions(i, lower_out_of_bounds) = particle_positions(i, lower_out_of_bounds) + ...
                    2 * (lower_bounds(lower_out_of_bounds) - particle_positions(i, lower_out_of_bounds));
                particle_positions(i, upper_out_of_bounds) = particle_positions(i, upper_out_of_bounds) - ...
                    2 * (particle_positions(i, upper_out_of_bounds) - upper_bounds(upper_out_of_bounds));
            else
                % Apply Final Boundary Adjustment
                particle_positions(i, :) = particle_positions(i, :) + particle_velocities(i, :);
                particle_positions(i, :) = min(max(particle_positions(i, :), lower_bounds), upper_bounds);
            end
    
            % Mutation operator
            if rand < mutation_probability
                gk_index = randi([1, num_assets]); % Randomly choose a variable index
                flip = randi([0, 1]); % Random event returning 0 or 1
            
                if flip == 0
                    mutation_amount = delta_fun(iter, upper_bounds(gk_index) - ...
                        particle_positions(i, gk_index), num_iterations, b);
                    mutated_gk = particle_positions(i, gk_index) + mutation_amount;
                else
                    mutation_amount = delta_fun(iter, particle_positions(i, gk_index) - ...
                        lower_bounds(gk_index), num_iterations, b);
                    mutated_gk = particle_positions(i, gk_index) + mutation_amount;
                end
            
                % Apply the mutation
                particle_positions(i, gk_index) = mutated_gk;
            end
    
            % Update positions within the cardinality constraint
            current_set = find(binary_positions(i, :));
            Knew = length(current_set);
    
            % Adding Assets: if Knew < num_selected_assets
            while Knew < num_selected_assets
                available_indices = setdiff(1:num_assets, current_set);
                new_asset = available_indices(randi(length(available_indices)));
                binary_positions(i, new_asset) = 1;
                particle_positions(i, new_asset) = epsilon; % Assign minimum proportional value
                current_set = find(binary_positions(i, :));
                Knew = length(current_set);
            end
    
            % Removing Assets: if Knew > num_selected_assets
            while Knew > num_selected_assets
                [~, sorted_indices] = sort(particle_positions(i, current_set));
                asset_to_remove = current_set(sorted_indices(1));
                binary_positions(i, asset_to_remove) = 0;
                particle_positions(i, asset_to_remove) = 0;
                current_set = find(binary_positions(i, :));
                Knew = length(current_set);
            end
    
            % Proportional Value Adjustment
            for j = current_set
                if particle_positions(i, j) < epsilon
                    particle_positions(i, j) = epsilon;
                elseif particle_positions(i, j) > delta
                    particle_positions(i, j) = delta;
                end
            end
    
            total_weight = sum(particle_positions(i, current_set));
            if total_weight > 0
                particle_positions(i, current_set) = particle_positions(i, current_set) / total_weight;
            end
            
            % Evaluate fitness
            %fitness = fitness_function(particle_positions(i, :), risk_aversion);
            fitness = compute_fitness(particle_positions(i, :), risk_aversion, mean_returns, cov_matrix, num_selected_assets, epsilon, delta, max_returns, max_variance, min_returns);
    
            % Update personal bests
            if fitness < personal_best_scores(i)
                personal_best_scores(i) = fitness;
                personal_best_positions(i, :) = particle_positions(i, :);
            end
    
            % Update global best
            if fitness < global_best_score
                global_best_score = fitness;
                global_best_position = particle_positions(i, :);
            end
        end
           
        % Check termination criterion
        if abs(prev_best_score - global_best_score) < 1e-6 % If no improvement in score
            num_iterations_without_improvement = num_iterations_without_improvement + 1;
        else
            num_iterations_without_improvement = 0; % Reset counter
            prev_best_score = global_best_score;
        end

        if num_iterations_without_improvement >= 100 % Terminate if no improvement for 100 iterations
            % Check the global_best_score 
            if (global_best_score > 10) % && num_restart < 100
                fprintf('Reinitializing parameters at iteration %d for lambda = %.2f\n', iter, risk_aversion_values(lambda));

                % Reinitialize all parameters
                particle_positions = rand(num_particles, num_assets);
                particle_positions = lower_bounds + particle_positions .* (upper_bounds - lower_bounds);
                particle_velocities = zeros(num_particles, num_assets);
                binary_positions = zeros(num_particles, num_assets);
        
                for ij = 1:num_particles
                    selected_indices = randperm(num_assets, num_selected_assets);
                    binary_positions(ij, selected_indices) = 1;
                end

                % personal_best_positions = particle_positions;
                % personal_best_scores = inf(num_particles, 1);
                % global_best_position = zeros(1, num_assets);
                % global_best_score = inf;

                % Reset the termination criterion
                num_iterations_without_improvement = 0;
                % num_restart = num_restart + 1;
                prev_best_score = inf;
        
            else
                fprintf('Terminating at iteration %d for lambda = %.2f\n', iter, risk_aversion_values(lambda));
                break;
            end
        end
    end

    % Calcolo del rischio e del rendimento per il miglior portafoglio
    portfolio_return = global_best_position * mean_returns;
    portfolio_variance = global_best_position * cov_matrix * global_best_position';
    PSO_portfolios(lambda, :) = [risk_aversion, portfolio_variance, portfolio_return, global_best_position];
    disp('Minimum score:');
    disp(global_best_score);

end

%% PSO Portfolios
PSO_variances = PSO_portfolios(2:end, 2);
PSO_returns = PSO_portfolios(2:end, 3);

writematrix(PSO_portfolios, 'data/PSO_portfolios.csv');

%% Plot
figure;
hold on;

% Plot efficient frontier
plot(efficient_frontier_risks, efficient_frontier_returns, 'b-', 'LineWidth', 2, 'DisplayName', 'Efficient Frontier');

% Plot PSO results
scatter(PSO_variances, PSO_returns, 50, 'r', 'filled', 'DisplayName', 'PSO Results');

xlabel('Portfolio Variance (Risk)');
ylabel('Portfolio Return');
title('Efficient Frontier, Results');
legend('Location', 'Best');
grid on;
hold off;


%%
% Fitness function 
function fitness = compute_fitness(weights, risk_aversion, mean_returns, cov_matrix, num_selected_assets, epsilon, delta, max_returns, max_variance, min_returns)
    variance = weights * cov_matrix * weights';
    returns = mean_returns' * weights';


% Fitness function 
    fitness = risk_aversion * variance/max_variance - (1 - risk_aversion) * (returns-min_returns)/(max_returns-min_returns);

    % Check if sum of weights exceeds 1
    if sum(weights) > 1
        fitness = fitness + 100* sum(weights);
    end
    % Check if the number of selected assets exceeds the cardinality constraint
    if nnz(weights) > num_selected_assets 
        fitness = fitness + 100 * abs(nnz(weights)-num_selected_assets);

    end
    % Check if any weight is outside the bounds [epsilon, delta]
    if any(nnz(weights) < epsilon) || any(weights > delta) 
        fitness = fitness + 100 * (sum(any(nnz(weights) < epsilon)) + sum(weights > delta));
    end

    % Check the returns constraints
    if returns > max_returns
         fitness = fitness + 100 * (1+returns/max_returns);
    end

    % Penalty for variance outside bounds
    if variance > max_variance 
        fitness = fitness + 100 * (1+variance/max_variance);
    end


end
