pm = imread('prob_map_uint.png');

mat_hmax = imhmax(pm, 125, 8);
py_hmax = imread('imhmaxima_result.png');

tmp = mat_hmax == py_hmax;
if all(tmp(:))
    disp('Matlab hmax same as python imhmax')
end

mat_extmax = imread('extended_maxima_result.png');
py_extmax = imregionalmax(mat_hmax, 8);
tmp = mat_extmax == py_extmax;
if all(tmp(:))
    disp('Matlab extendex max same as python extended max')
end