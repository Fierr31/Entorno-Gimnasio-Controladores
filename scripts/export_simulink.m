%% Exporta la simulación de Simulink al formato que espera validate_vs_simulink.py
% 1. Corre tu modelo con EXACTAMENTE el escenario de config/quad_params.yaml
%    (validation: x0, reference, t_final; timing: dt_sim, dt_ctrl, loop_delay).
% 2. Registra el vector de 16 estados en un "To Workspace" (Timeseries) llamado x_log,
%    en el orden: x y z vx vy vz phi theta psi p q r T1 T2 T3 T4.
% 3. Ejecuta este script. Ajusta nombres de modelo y variables a los tuyos.

mdl = 'quad_lqr';              % TODO: nombre de tu modelo
ctrl = 'lqr';                  % 'lqr' o 'lqi'
out = sim(mdl);

ts = out.x_log;                % TODO: nombre de tu señal registrada
t  = ts.Time(:);
X  = squeeze(ts.Data);
if size(X,1) == 16 && size(X,2) ~= 16, X = X.'; end   % asegurar N x 16

T = array2table([t X], 'VariableNames', ...
    {'t','x','y','z','vx','vy','vz','phi','theta','psi','p','q','r','T1','T2','T3','T4'});
writetable(T, sprintf('simulink_%s.csv', ctrl));

% Ganancias para la verificación previa (K: 4x16 LQR, o [Kx Ki]: 4x20 LQI)
writematrix(K, sprintf('K_%s_matlab.csv', ctrl));   % TODO: nombre de tu variable de ganancia
fprintf('Listo: simulink_%s.csv y K_%s_matlab.csv -> copia ambos a data/\n', ctrl, ctrl);
