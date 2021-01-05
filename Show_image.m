lb= load('20.mat');
lu= load('epi/papillon/Seg_u.mat');
lv= load('epi/papillon/Seg_v.mat');
colors = rand( 2000, 3);
imshow( label2rgb(lb.X(:, :,5,5 ), colors) );
