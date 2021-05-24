d= uigetdir(pwd, 'Select a folder');
files = dir(fullfile(d, '*.tif'));


%%
I = imread('data\stdev-images\Subject27_Session236_OS_(5,0)_1x1_1161_OA790nm_dewarped1_extract_reg_std.tif');
I = uint8(normalizeValues(I, 0, 255));
imshow(I);

BW = imextendedmax(I, 120);
save('BW.mat', 'BW')
%%
I = imread('data\stdev-images\Subject27_Session236_OS_(2,0)_1x1_1158_OA850nm_dewarped1_extract_reg_std.tif');
In = uint8(normalizeValues(I, 0, 255));
imshow(In);

%%
BW = imhmax(In, 208);
imshow(BW, []);

%%
I = imread('data\stdev-images\Subject27_Session236_OS_(0,-2)_1x1_1154_OA850nm_dewarped1_extract_reg_std.tif');
I = uint8(normalizeValues(I, 0, 255));
imshow(I);

%%
H = 100;

BW = imhmax(I, H, 4);
imshow(BW)
%%
BW = imextendedmax(I, 100);
imshow(BW);