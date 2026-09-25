fs = 250;
window = fs;
noverlap = fs/2;
nfft = 250;
[s, f, t, p] = spectrogram(Data, window, noverlap, nfft, fs);

alpha_mask = (f >= 8 & f <= 12);
alpha_power_over_time = mean(p(alpha_mask, :), 1);

closed_starts = Events(1:2:end);
closed_ends   = Events(2:2:end); 

opened_starts = [0, reshape(closed_ends, 1, [])];   % Appending the final time of the recording
opened_ends   = [reshape(closed_starts, 1, []), t(end)];    % Appending the initial timo of the recording

% Minimum length for safe iteration when comparing closed/opened events
len_min = min([size(closed_starts,2) size(closed_ends,2) size(opened_starts,2) size(opened_ends,2)]);

closed_power_vals = [];
opened_power_vals = [];

% Extract all Alpha power values during "Eyes Closed" (Signal)
for i = 1:len_min
    % Find indices in 't' that fall within this specific closed interval
    idx = (t >= closed_starts(i)) & (t <= closed_ends(i));
    closed_power_vals = [closed_power_vals, alpha_power_over_time(idx)];
end

% Extract all Alpha power values during "Eyes Opened" (Noise Baseline)
for i = 1:length(opened_starts)
    idx = (t >= opened_starts(i)) & (t <= opened_ends(i));
    opened_power_vals = [opened_power_vals, alpha_power_over_time(idx)];
end

% Calculate the global means
P_signal = mean(closed_power_vals);
P_noise  = mean(opened_power_vals);

% Signal-to-Noise Ratio
SNR_linear = P_signal / P_noise;
SNR_dB = 10 * log10(SNR_linear);

%% --- Results ---
fprintf('\n--- SNR Analysis ---\n');
fprintf('Mean Alpha Power (Eyes Closed - Signal): %.4f\n', P_signal);
fprintf('Mean Alpha Power (Eyes Opened - Noise):  %.4f\n', P_noise);
fprintf('SNR (Linear): %.2f\n', SNR_linear);
fprintf('SNR (dB):     %.2f dB\n', SNR_dB);

%% --- Plotting ---

figure;
hold on;
hLine = plot(t, alpha_power_over_time, 'b', 'LineWidth', 1.5, 'DisplayName', 'Alpha Power (8-12 Hz)');
yl = ylim; 


len_plot = max([length(closed_starts), length(closed_ends), length(opened_starts), length(opened_ends)]); % Used to cover the max length

for i = 1:len_plot
    
    % Draw Opened Patch (Gray)
    if i <= length(opened_starts) && i <= length(opened_ends)
        if opened_ends(i) > opened_starts(i)
            x_box_opened = [opened_starts(i), opened_ends(i), opened_ends(i), opened_starts(i)];
            y_box = [yl(1), yl(1), yl(2), yl(2)];
            
            hPatchOpened = patch(x_box_opened, y_box, [0.8 0.8 0.8], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
            uistack(hPatchOpened, 'bottom');
            
            if i == 1
                hPatchOpened.DisplayName = 'Eyes Opened';
            else
                hPatchOpened.HandleVisibility = 'off';
            end
        end
    end
    
    % Draw Closed Patch (Purple)
    if i <= length(closed_starts) && i <= length(closed_ends)
        x_box_closed = [closed_starts(i), closed_ends(i), closed_ends(i), closed_starts(i)];
        hPatchClosed = patch(x_box_closed, y_box, [0.8 0.3 0.8], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        uistack(hPatchClosed, 'bottom'); 
        
        if i == 1
            hPatchClosed.DisplayName = 'Eyes Closed';
        else
            hPatchClosed.HandleVisibility = 'off'; 
        end
    end
end

xlabel('Time (seconds)');
ylabel('Alpha Power Spectral Density');
legend('show', 'Location', 'northeast');
hold off;
