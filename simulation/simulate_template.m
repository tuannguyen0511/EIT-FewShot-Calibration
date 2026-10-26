% Template for generating simulated training pairs for a 16-electrode tank (EIDORS).
% This is a guide, not the exact generator used in the paper. Set the geometry,
% inclusion distribution and noise model to match your own instrument.
% Requires EIDORS (http://eidors3d.sourceforge.net), run eidors_startup first.

n_samples   = 1000;            % number of simulated cases
tank_radius = 1;               % normalised radius (scale to your tank)
n_elec      = 16;
% ---- user-defined inclusion model (choose for your application) ----
n_incl_range   = [1 3];        % number of inclusions per case
radius_range   = [0.08 0.30];  % inclusion radius (fraction of tank radius)
sigma_resistive = [];          % e.g. [low high] relative to background; SET THIS
sigma_conductive = [];         % e.g. [low high] relative to background; SET THIS
snr_db = 60;                   % additive measurement noise
assert(~isempty(sigma_resistive) && ~isempty(sigma_conductive), 'Set the conductivity ranges first.');

% ---- forward model: 2-D circular tank, complete electrode model ----
fmdl = ng_mk_cyl_models(0, [n_elec], [0.05]);           % 2-D, electrode width as needed
[stim, meas_sel] = mk_stim_patterns(n_elec, 1, [0 1], [0 1], {'no_meas_current'}, 1);  % adjacent: 208 values
fmdl.stimulation = stim;
img_bg = mk_image(fmdl, 1);
v_ref = fwd_solve(img_bg);

% ---- pixel grid for labels (64 x 64 over the unit disk) ----
[px, py] = meshgrid(linspace(-1 + 1/64, 1 - 1/64, 64));
ctr = interp_mesh(fmdl);                                  % element centres

V = zeros(numel(v_ref.meas), n_samples); L = zeros(64, 64, n_samples);
for s = 1:n_samples
    img = img_bg; lab = ones(64);
    for q = 1:randi(n_incl_range)
        r = radius_range(1) + rand * diff(radius_range);
        rho = rand * (0.9 - r); th = 2 * pi * rand; cx = rho * cos(th); cy = rho * sin(th);
        if rand < 0.5, rng_s = sigma_resistive; else, rng_s = sigma_conductive; end
        val = exp(log(rng_s(1)) + rand * (log(rng_s(2)) - log(rng_s(1))));
        in_e = (ctr(:,1) - cx).^2 + (ctr(:,2) - cy).^2 < r^2;
        img.elem_data(in_e) = val;
        lab((px - cx).^2 + (py - cy).^2 < r^2) = val;
    end
    v = fwd_solve(img);
    dv = v.meas - v_ref.meas;
    dv = add_noise(snr_db, dv, v_ref.meas);                % or your own noise model
    V(:, s) = dv; L(:, :, s) = lab;
end
save('simulated_pairs.mat', 'V', 'L', '-v7.3');
% Sensitivity operator for the J-map input: Jacobian at the homogeneous background,
% mapped from mesh elements to the 64 x 64 pixel grid (see the paper, Section III-A).
