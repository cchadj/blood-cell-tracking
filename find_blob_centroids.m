avg_790_sig_gauss_blur = imread('tmp\AVG_Stack_790nm_2sigGuass_maxPx.tif'); 
avg_850_sig_gauss_blur = imread('tmp\AVG_Stack_850nm_2sigGuassBlur_maxPx.tif'); 

avg_cell_790m  = imread('tmp\AVG_Stack_790nm.tif');
avg_cell_850m  = imread('tmp\AVG_Stack_850nm.tif');

avg_cell_790py = imread('avg-cell-OA790.png');
avg_cell_850py = imread('avg-cell-OA850.png');

imlist = {};
imlist{1} = avg_cell_790m;
imlist{2} = avg_cell_850m;
imlist{3} = avg_cell_790py;
imlist{4} = avg_cell_850py;

% Note, Matlab average images and python average images only different by 
% 2 grayscale levels at most.
avg_cell_790_diff = abs(single(avg_cell_790m(:)) - single(avg_cell_790py(:)));
assert(all(avg_cell_790_diff <= 3));

avg_cell_850_diff = abs(single(avg_cell_850m(:)) - single(avg_cell_850py(:)));
assert(all(avg_cell_850_diff <= 2));

for i = 1:size(imlist, 2)
    subplot(2, 2, i)
    imshow(imlist{i});
end
%%
correlation_output = 'cell-patches-ouptut';
S = dir(fullfile(correlation_output, 'cell-patch*OA850*.png')); % pattern to match filenames.
cell_patches_850 = zeros(numel(S), 75, 75);
for k = 1:numel(S)
    F = fullfile(correlation_output,S(k).name);
    I = imread(F);
    imshow(I)
    cell_patches_850(k, :, :) = I;
end

S = dir(fullfile(correlation_output, 'cell-patch*790*.png')); % pattern to match filenames.
cell_patches_790 = zeros(numel(S), 75, 75);
for k = 1:numel(S)
    F = fullfile(correlation_output,S(k).name);
    I = imread(F);
    imshow(I)
    cell_patches_790(k, :, :) = I;
end
%%
avg_cell_patch_790_m = uint8(squeeze(mean(cell_patches_790, 1)));
avg_cell_patch_850_m = uint8(squeeze(mean(cell_patches_850, 1)));
figure
subplot(1, 2, 1)
imshow(avg_cell_patch_790_m)
subplot(1, 2, 2)
imshow(avg_cell_patch_850_m)

avg_cell_790_diff = abs(single(avg_cell_790m(:)) - single(avg_cell_patch_790_m(:)));
assert(all(avg_cell_790_diff <= 1));

avg_cell_850_diff = abs(single(avg_cell_850m(:)) - single(avg_cell_patch_850_m(:)));
assert(all(avg_cell_850_diff <= 1));

%%
original_image = avg_790_sig_gauss_blur;
original_image = uint8(original_image);

blurred_image = imgaussfilt(original_image, 0);
binary_image = imregionalmax(blurred_image);
    
[labeled_image, n_regions] = bwlabel(binary_image);

region_properties = regionprops(labeled_image, original_image, 'all');

all_region_centroids = [region_properties.Centroid];
centroids_x = all_region_centroids(1:2:end-1);
centroids_y = all_region_centroids(2:2:end);
region_intensities = [region_properties.MeanIntensity];

region_areas = [region_properties.Area];

regions_to_discard_area_mask = region_areas < mean(region_areas) - 0.1 * std(region_areas);
regions_to_discard_intesity_mask = region_intensities < mean(region_intensities) - 0.1 *  std(region_intensities);

regions_to_discard_mask = regions_to_discard_area_mask | regions_to_discard_intesity_mask;

centroids_x(regions_to_discard_mask) = [];
centroids_y(regions_to_discard_mask) = [];
region_intensities(regions_to_discard_mask) = [];

[~, max_region_intensity_idx] = maximum(region_intensities, 1);
hold on;

subplot(1, 3, 1);
imshow(original_image);
hold on;
scatter(centroids_x(max_region_intensity_idx), centroids_y(max_region_intensity_idx));

subplot(1, 3, 2);
imshow(blurred_image);

subplot(1, 3, 3);
imshow(binary_image);
hold on;
scatter(centroids_x(max_region_intensity_idx), centroids_y(max_region_intensity_idx));
%%
clear all;
clf;
avg_cell_790py = imread('avg-cell-OA790.png');
avg_cell_850py = imread('avg-cell-OA850.png');
    
I = avg_cell_790py;
% Get the size of the image
[rows, columns, numberOfColorChannels] = size(I);
% Determine starting and ending rows and columns.
template_h = 21;
template_w = 21;
row1 = floor(rows / 2 - 10 );
col1 = floor(columns / 2 - 10);
% Extract sub-image using imcrop():
template = imcrop(I, [col1, row1, 20, 20]);
[template_h, template_w] = size(template);

correlation_output = normxcorr2(template, avg_cell_850py);
[~, max_idx] = max(abs(correlation_output(:)));
[y_peak, x_peak] = ind2sub(size(correlation_output), max_idx(1));
% Because cross correlation increases the size of the image, 
% we need to shift back to find out where it would be in the original image.
corr_offset = [(x_peak-size(template,2)) (y_peak-size(template,1))];
box_rect = [corr_offset(1) corr_offset(2) template_h template_w];

subplot(1, 2, 1);
imshow(template);

subplot(1, 2, 2);
imshow(I);
hold on;
rectangle('position', box_rect, 'edgecolor', 'g', 'linewidth',2);
% Give a caption above the image.
title('Template Image Found in Original Image', 'FontSize', 10);

[~, idx] = max(abs(correlation_output(:)));
[y, x] = ind2sub(size(I), idx(1));

% subplot(1, 3, 1);
% imshow(I);
% hold on;
% scatter(x, y);
% 
% subplot(1, 3, 2);
% imshow(template);
% hold on;
% scatter(x, y);
% 
% subplot(1, 3, 3);
% imshow(correlation_output, []);


%%
I = original_image;
[~, threshold] = edge(I, 'sobel');
fudgeFactor = 0.5;
BWs = edge(I,'sobel',threshold * fudgeFactor);
edge(I,'sobel',threshold * fudgeFactor);
imshow(BWs)
title('Binary Gradient Mask')

%%
se90 = strel('line',3,90);
se0 = strel('line',3,0);
BWsdil = imopen(BWs, se90 );
BWsdil = imopen(BWsdil,8,se90 );
BWsdil = imopen(BWsdil, se90 );
BWsdil = imopen(BWsdil, se90 );

imshow(BWsdil)
title('Dilated Gradient Mask');

%%
BWdfill = imfill(BWsdil,'holes');
imshow(BWdfill)
title('Binary Image with Filled Holes');

%%
BWnobord = imclearborder(BWdfill,4);
imshow(BWnobord)
title('Cleared Border Image');

%%
I1 = avg_cell_790py;
I2 = avg_cell_850py;

mthresh = 0;
points1 = detectSURFFeatures(I1, 'MetricThreshold', mthresh);
points2 = detectSURFFeatures(I2, 'MetricThreshold', mthresh);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2, 'montage');
legend('matched points 1','matched points 2');

%%
I1 = template;
% I1 = avg_cell_790py;
I2 = avg_cell_850py;

points1 = detectFASTFeatures(I1, 'MinQuality', 1, 'MinContrast', 0.0001);
points2 = detectFASTFeatures(I2, 'MinQuality', 1, 'MinContrast', 0.0001);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2, 'montage');
legend('matched points 1','matched points 2');

%%
I1 = avg_cell_790py;
I2 = avg_cell_850py;

points1 = detectBRISKFeatures(I1, 'MinContrast', 0.0001, 'NumOctaves', 5);
points2 = detectBRISKFeatures(I2, 'MinContrast', 0.0001, 'NumOctaves', 5);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2, 'montage');
legend('matched points 1','matched points 2');
%%
imshow(labeloverlay(I,BWfinal))
title('Mask Over Original Image')

function [val, idx] = maximum(data, order)
    
    [val, idx] = max(data(:));
    for i = 2:order
        data(idx) = [];
        [val, idx] = max(data(:));
    end
end

