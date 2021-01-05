

%%
%% Generate *View Consistent Light Field Superpixels* for the 
%% input light field  using the parameter settings in param
%%

colors = rand( 2000, 3);
addpath('./prop', './lines', './seg', './util');%검색폴더 경로 추가
D = '/home/jo/PycharmProjects/pythonProject2/additional/antinous/antinous.h5'
%D='resize.h5'
param = parameters;
cluster = parcluster('local');
cluster.NumWorkers = param.nWorkers;%병렬처리
saveProfile(cluster);
% LF = HCIloadLF( D, 'lab');
% % 
LFs = HCIloadLF( D, 'lab');
LF=zeros(size(LFs));
    for i = 1:9
        for j=1:9
        asd=abs(10-j);
        LF(:,:,:,i,j)=LFs(:,:,:,i,asd);
        %imshow(LFs(:,:,:,i,asd));
        %title('disagree Map');
        %  waitforbuttonpress;
        end
    end
szLF = [size(LF, 1) size(LF, 2) size(LF, 4) size(LF, 5)]; % light field size in y, x, v, u
szEPIu = [szLF(4) szLF(2) szLF(1)]; % EPI size in u...
szEPIv = [szLF(3) szLF(1) szLF(2)]; %... and v directions
cviewIdx = ceil(szLF(3)/2);% 9--> 5? 올림 center view index
LFuc = squeeze(LF(:, :, :, cviewIdx, :));% u slicing    512   512     3     9

  % Filter the views to remove noise
  % Each channel is filtered separately (using imgaussfilt on an 3-channel image
  % doesn't give the same results as this, for some reason).
for i = 1:size(LFuc, 4)% 9
  LFuc(:, :, :, i) = imgaussfilt( squeeze(LFuc(:, :, :, i)), 0.85);    
end
  EPIuc = permute(LFuc, [4 2 3 1]);%    9   512     3   512(u,x,color,y)
%size(EPIuc)



  % Get the central column of LF images, and their EPIs
  LFvc = squeeze(LF(:, :, :, :, cviewIdx));
  %size(LFvc)
  LFvc = permute(LFvc, [2 1 3 4]);%xyuv
  %size(LFvc)
  for i = 1:size(LFvc, 4) 
    LFvc(:, :, :, i) = imgaussfilt( squeeze(LFvc(:, :, :, i)), 0.85);
  end
  EPIvc = permute(LFvc, [4 2 3 1]);%9   512     3   512(v,y,color,x)
  %size(EPIvc)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  % EDGE DETECTION & LINE FITTING  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Lv = {}; Lu = {};
  segMinWidth = round(const.SegMinWidthMultiplier * szLF(3));%2 ??

  for i = param.linesPyramidLevels - 1:-1:0 %% 1, 0
    disp(i);
    % Scale-up results from lower pyramid level 2
    LvScaled = cellfun(@(l) l .* (2), repelem(Lv, 2, 1), 'UniformOutput', false);%.*배열연산
    LuScaled = cellfun(@(l) l .* (2), repelem(Lu, 2, 1), 'UniformOutput', false);%cell형태로 값을 반환
    % Fit lines at current level...
%     %%%%%%%%%%%%%%lines%%%%%%%%%%%%%%%
%   
%       % Resize the lightfield to the desired pyramid level (in the spatial domain)
%   w = floor(size(LFvc, 2) / (2^i));
%   h = floor(size(LFvc, 1) / (2^i));
%   d = size(LFvc, 4);
% 
%   if i ~= 0
%     LFvc = imresize(LFvc, [h w]);
%   end
%   EPI = permute(LFvc, [4 2 3 1]);
% 
%   % Generate the filters for edge detection, ...
%   filterCount = ceil((param.maxAbsDisparity * d) / (2^i));
%   [F, gx, gy] = genFilters( filterCount, [d w]);
% 
%   % ... and the line templates for line fitting
%   [T, m] = genLines( filterCount, [d w]);
% 
%   % Calculate the edge confidence and slope maps
%   [Ev, Z] = epis2edges( EPI, F, gx, gy);
% 
%   % Fit lines to the edges
%   [Lv, Lc] = edges2lines(Ev, Z, T, m);
% 
%   % Remove outliers
%   Lv = filterOutliers(Lv, Lc, [h, w, d], i);
%     
%    
%       % Resize the lightfield to the desired pyramid level (in the spatial domain)
%   w = floor(size(LFuc, 2) / (2^i));
%   h = floor(size(LFuc, 1) / (2^i));
%   d = size(LFuc, 4);
% 
%   if i ~= 0
%     LFuc = imresize(LFuc, [h w]);
%   end
%   EPI = permute(LFuc, [4 2 3 1]);
% 
%   % Generate the filters for edge detection, ...
%   filterCount = ceil((param.maxAbsDisparity * d) / (2^i));
%   [F, gx, gy] = genFilters( filterCount, [d w]);
% 우리의 실험은 철저하지만 acaveat와 함께 제공됩니다. GC10 및 TP09와 같은 특정 슈퍼 픽셀 방법은 색상 정보를 고려하지 않는 반면 다른 방법은 고려합니다. 이로 인해 성능이 저하 될 수 있습니다.
%   % ... and the line templates for line fitting
%   [T, m] = genLines( filterCount, [d w]);
% 
%   % Calculate the edge confidence and slope maps
%   [Eu, Z] = epis2edges( EPI, F, gx, gy);
% 
%   % Fit lines to the edges
%   [Lu, Lc] = edges2lines(Eu, Z, T, m);
% 
%   % Remove outliers
%   Lu = filterOutliers(Lu, Lc, [h, w, d], i);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % Fit lines at current level...
    [Lv, Ev] = lines(LFvc, i, param);
    [Lu, Eu] = lines(LFuc, i, param);

    Lv = mergeLines(Lv, LvScaled, true, segMinWidth); 
     Lu = mergeLines(Lu, LuScaled, true, segMinWidth);
  end
  %L= outlier regjection  E= Confidence
%     disparityRange = [-2 2];
%     ss=squeeze(Eu(1,:,:));
%     imshow(ss,disparityRange);
%     title('Disparity Map');
%     colormap(gca,jet) 
%     colorbar

     

  % Add the vertical line detections to the horizontal ones, and vice versa, for 
  % increased detection accuracy
  Lv = mergeLines(Lv, sparsifyLines(hor2vert(Lu, szLF), segMinWidth), true, segMinWidth);
  Lu = mergeLines(Lu, sparsifyLines(vert2hor(Lv, szLF), segMinWidth), true, segMinWidth);%2point line clear
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  % OCCLUSION-AWARE EPI SEGMENTATION %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [Mv, Sv] = lines2matches(Lv, Ev, szEPIv );
  [Mu, Su] = lines2matches(Lu, Eu, szEPIu );

dMapv = permute( segs2disparity(Sv, szEPIv), [3 2 1] );
dMapv = medfilt2(dMapv(:, :, cviewIdx), [10 1]);
% dMapu = permute( segs2disparity(Su, szEPIu), [3 2 1] );
% dMapu = medfilt2(dMapu(:, :, cviewIdx), [10 1]);
%   Seg_u= zeros(szLF(1),szLF(2),szLF(4));
%   Seg_v= zeros(szLF(1),szLF(2),szLF(3));
%   
  Seg_u=mask_gen(Mu, Lu, dMapv', EPIuc);
  Seg_v=mask_gen(Mv, Lv, dMapv, EPIvc);
  Seg_v = permute(Seg_v,[2 1 3]);
  
  epi_gen=zeros(9,512);
    for i=1:9
            epi_gen(i,:)=Seg_u(3,:,i);
    end
        
  save(['/home/jo/lightfieldsuperpixels/custom/Seg_u.mat'], 'Seg_u');
  save(['/home/jo/lightfieldsuperpixels/custom/Seg_v.mat'], 'Seg_v');
%     disparityRange = [-2 2];
%     imshow(dMapu,disparityRange);
%     title('Disparity Map');
%     colormap(gca,jet) 
%     colorbar

%     disp('hi')
%     imshow(dMapv',disparityRange);
%     title('Disparity Map2');
%     colormap(gca,jet) 
%     colorbar
  [Su, Lu] = matches2Features(Mu, Lu, dMapv', EPIuc);% i think,,, it is,,, not good? isn't it ? cuz dmap` something wrong,,,,
  [Sv, Lv] = matches2Features(Mv, Lv, dMapv, EPIvc);%Sv= 4point segment set , epiindex, Labcolor n*8
  
  for szSP = param.szSuperpixels

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % SPATIO-ANGULAR SEGMENTATION VIA CLUSTERING %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % We use the term "segment" for the features of a pair of matched lines
    %[labelsu, labelsv] = segs2clusters(Sv, Su, szLF, szSP, param.wxy, param.wz, param.wlab); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  wxy=param.wxy;
  wz=param.wz;
  wlab=param.wlab;
  % Count the total number of segments in the horizontal and 
  % vertical segmentations. Each segment is a point in the clustering
  n = zeros(2, 1);
  n(1) = sum(cellfun(@(s) size(s, 1), Sv));
  n(2) = sum(cellfun(@(s) size(s, 1), Su));

  % Get the point features from the vertical and horizontal segments
  % Using separate arrays for the point coordinates showed better performance
  px = zeros( n(1) + n(2), 1);
  py = zeros( n(1) + n(2), 1);
  pz = zeros( n(1) + n(2), 1);
  pl = zeros( n(1) + n(2), 1);
  pa = zeros( n(1) + n(2), 1);
  pb = zeros( n(1) + n(2), 1);

  s = cell2mat(Sv);
  pz  (1:n(1))    = (s(:, 2) - s(:, 1)) ./ szLF(3);% 수직인 직선은 depth가 0이 되긴하는데 거의 수직인 직선을 찾는뎅유?
  % For the vertical direction, x and y are flipped...
  py (1:n(1), :) = [(s(:, 2) - s(:, 1)) ./ 2 + (s(:, 1) + s(:, 3)) ./ 2];
  px (1:n(1), :) = [s(:, 5)];
  pl (1:n(1), :) = [s(:, 6)];
  pa (1:n(1), :) = [s(:, 7)];
  pb (1:n(1), :) = [s(:, 8)];

  s = cell2mat(Su);
  pz (n(1) + 1:n(2) + n(1))    = (s(:, 2) - s(:, 1)) ./ szLF(4);
  px (n(1) + 1:n(2) + n(1), :) = [(s(:, 2) - s(:, 1)) ./ 2 + (s(:, 1) + s(:, 3)) ./ 2];%왜 이게 center야?
  py (n(1) + 1:n(2) + n(1), :) = [s(:, 5)];
  pl (n(1) + 1:n(2) + n(1), :) = [s(:, 6)];
  pa (n(1) + 1:n(2) + n(1), :) = [s(:, 7)];
  pb (n(1) + 1:n(2) + n(1), :) = [s(:, 8)];

  % Normalize the depth values
  pz = pz + abs(min(pz));
  pz = pz ./ max(pz);

  % Initialize the cluster centers
  % Cluster centers are created on a regular grid in the spatial domain
  % and then assigned the value of the closest point
  %
  [cx, cy, cz] = meshgrid( szSP/2: szSP: szLF(2) - szSP/2, ...
			   szSP/2: szSP: szLF(1) - szSP/2, ...
			   0);
  cx = cx(:);
  cy = cy(:);
  cz = cz(:);
  cl = zeros(size(cx));
  ca = zeros(size(cx));
  cb = zeros(size(cx));

  nLabels = size(cx, 1);
  labels = zeros( n(1) + n(2), 1);

  parfor i = 1:nLabels  % cluster center
    dxy = (px - cx(i)).^2 + (py - cy(i)).^2;
    [~, minIdx] = min(dxy);
    
    cx(i) = px(minIdx);
    cy(i) = py(minIdx);
    cz(i) = pz(minIdx);
    cl(i) = pl(minIdx);
    ca(i) = pa(minIdx);
    cb(i) = pb(minIdx);
  end

  % K-Means clustering
  %
  for j = 1:const.KmeansIterations
    parfor i = 1:n(1) + n(2)
      dxy  = sqrt((cx - px(i)).^2 + (cy - py(i)).^2);
      dlab = sqrt((cl - pl(i)).^2 + (ca - pa(i)).^2 + (cb - pb(i)).^2);
      dz   = abs(cz - pz(i));

      d = wxy .* dxy + wz .* dz + wlab .* dlab;  %compactness test
      [~, minIdx] = min(d(:));
      labels(i) = minIdx;
    end

    parfor i = 1:nLabels
      idx = find(labels == i);
      cx(i) = mean(px(idx));
      cy(i) = mean(py(idx));
      cz(i) = mean(pz(idx));
      cl(i) = mean(pl(idx));
      ca(i) = mean(pa(idx));
      cb(i) = mean(pb(idx));
    end
  end

  labelsv = mat2cell(labels(1:n(1)), cellfun(@(s) size(s, 1), Sv), 1);
  labelsu = mat2cell(labels(n(1) + 1: n(1) + n(2)), cellfun(@(s) size(s, 1), Su), 1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Add the label as an additional feature to both vertical and horizontal segments
    % Also, the color features are not required after the clustering stage so they can 
    % be removed to minimize the amount of data moved about
    Su = cellfun(@(s, lbl) [s(:, 1:5) lbl], Su, labelsu, 'UniformOutput', false); 
    Sv = cellfun(@(s, lbl) [s(:, 1:5) lbl], Sv, labelsv, 'UniformOutput', false);
    
    % Project the vertical segments into all off-center columns, and the horizontal segments 
    % into all off-center rows. 
    %
    % After doing this, we'll have two sets of labels (U and V) for each light field view: 
    % one each from the horizontal and vertical segmentations.
    %
    U = zeros(szLF);
    V = zeros(szLF);
    parfor i = [1:cviewIdx cviewIdx + 1:szLF(4)] %%혹시 얘 때문에~~~???~~~
      viewOffset = cviewIdx - i;
      %disp(viewOffset)
      U(:, :, i, :) = permute( segs2labels( segsReproject(Su, -viewOffset, szLF),...
					   szEPIu), [3 2 1]);
      V(:, :, :, i) = permute( permute( segs2labels( segsReproject(Sv, -viewOffset, szLF), ...
						    szEPIv), [3 2 1]), [2 1 3]);
    end
%     for i = 1:9
%         for j=1:9
%         imshow(label2rgb(U(:, :,i,j ), colors));
%         title('U');
%          waitforbuttonpress;
%         imshow(label2rgb(V(:, :,i,j ), colors));
%         title('V');
%        
%         waitforbuttonpress;
%         end
%     end
    save(['/home/jo/lightfieldsuperpixels/custom/cl_U.mat'], 'U');
    save(['/home/jo/lightfieldsuperpixels/custom/cl_V.mat'], 'V');
  %%%%%%%%%%%%%%%%%%%%%
  % LABEL PROPAGATION %
  %%%%%%%%%%%%%%%%%%%%%

    % Find the regions with similar labels in the U and V segmentations for each view.
    %
    % Any pixel without a label (value == 0) in either U, or V is probably being disoccluded.
    % For such pixels we always use the label value from the other set.
    %
    X = zeros(szLF);
    for i = 1:szLF(4)
      parfor j = 1:szLF(3)

        % Median filtering to remove single pixel wide outshoot regions
        vi = medfilt2( V(:, :, j, i), [1 3]);
	ui = medfilt2( U(:, :, j, i), [3 1]);

	% Find pixels that are unlabelled (value == 0) in one set but not in the other
	vIdx = vi == 0;
	uIdx = ui == 0;

	vi(vIdx) = ui(vIdx);
	ui(uIdx) = vi(uIdx);
	X(:, :, j, i) = vi .* (vi == ui);
 
      end
    end

%     for i = 1:9
%         for j=1:9
%         imshow(label2rgb(X(:, :,i,j ), colors));
%         title('disagree Map');
%        
%         waitforbuttonpress;
%         end
%     end
     save(['/home/jo/lightfieldsuperpixels/custom/Disagree.mat'], 'X');
    % Disparity estimated from the edge line slopes of the angular segments is
    % used in the propagation of labels
    D = ones(szLF);
    Du=permute( segs2disparity(Su, szEPIu), [3 2 1] );
    Dv=permute(permute( segs2disparity(Sv, szEPIv), ...
	[3 2 1] ), [2 1 3]);
    D(:, :, :, cviewIdx) = permute( segs2disparity(Su, szEPIu), [3 2 1] );
    D(:, :, cviewIdx, :) = permute(permute( segs2disparity(Sv, szEPIv), ...
	[3 2 1] ), [2 1 3]);
     save(['/home/jo/lightfieldsuperpixels/custom/Disp_u.mat'], 'Du');
     save(['/home/jo/lightfieldsuperpixels/custom/Disp_v.mat'], 'Dv');
    disparityRange = [-2 2];
    for i = 1:9

        imshow(D(:,:,i,5),disparityRange);
        title('Disparity Map');
        colormap(gca,jet) 
        colorbar
        waitforbuttonpress;
     
    end
    
    
    % Light field view indices starting from the center outward, on the central
    % crosshair of views:
    %           :
    %           6 
    %           2 
    %    ...7 3 0 1 5 ...
    %           4
    %           8
    %           :
   
    theta = [0:pi/2:2*pi*(cviewIdx - 1) - pi/2];
    v = round([cviewIdx cviewIdx + sin(theta) .* repelem([1:cviewIdx - 1], 1, 4)]);
    u = round([cviewIdx cviewIdx + cos(theta) .* repelem([1:cviewIdx - 1], 1, 4)]);

    % Propagate the labels within each "crosshair" view and then project into
    % *all* other views
    for i = 1:length(v)
      X = propagate( X, LF, D(:, :, v(i), u(i)), [v(i) u(i)], szLF);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get the indices of the pixels which are unlabeled in the sparse label map
%       labelsSparse=X;
%    viewDisp=D(:, :, v(i), u(i));
%    viewIdx=[v(i) u(i)];
%       
%   viewSparseLbl = labelsSparse(:, :, viewIdx(1), viewIdx(2));
%   [Y, X] = find( viewSparseLbl <= 0 );
% 
%   Label propagation uses the color, spatial proximity, and disparity estimate
%   of neighborhood pixels to label an unlabelled pixel...
%   viewColor = imguidedfilter(LF(:, :, :, viewIdx(1), viewIdx(2)), ...
% 			     'NeighborhoodSize', const.PropGuidedFilterNhoodSize, ...
% 			     'DegreeOfSmoothing', const.PropGuidedFilterDegSmoothing);
%   imshow(viewColor);
%   viewDisp( viewSparseLbl == 0 ) = inf;%unlabeled
%    
%   w = szLF(2);
%   h = szLF(1);
%        
%   for k = 1:size(Y, 1)%unlabeled 횟수
%     x = X(k); 
%     y = Y(k);
% 
%     top, bottom, left, right of neighborhood window
%     nt = max(1, y - const.PropWinSz); 
%     nb = min(h, y + const.PropWinSz);
%     nl = max(1, x - const.PropWinSz);
%     nr = min(w, x + const.PropWinSz);
% 
%     nhood = viewSparseLbl(nt:nb, nl:nr);
%     [lbly, lblx] = find(nhood > 0);
%     if isempty( lbly )
%       continue;
%     end
% 
%     Get the image space indices of the labeled pixels in the neighborhood
%     idx = sub2ind([h w], lbly + nt - 1, lblx + nl - 1);
% 
%     z = viewDisp(idx);
%     [zu, ~, zuIdx] = unique(z);
%     dz = crossViewDistance(viewIdx(1), viewIdx(2), x, y, zu, LF, szLF);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     [~, zminIdx] = min(dz);
%     dz = dz(zuIdx);
%     zmin = zu(zminIdx);
% 
%     dlab = (viewColor(y, x, 1) - viewColor(idx)).^2 + ...
% 	   (viewColor(y, x, 2) - viewColor(idx + w * h)).^2 + ...
% 	   (viewColor(y, x, 3) - viewColor(idx + w * h * 2)).^2;
%     dxy = (lblx - size(nhood, 2)./2 ).^2 + (lbly - size(nhood, 1)./2 ).^2;
%     d = const.PropWlab .* dlab + ...
% 	const.PropWxy  .* dxy  + ...
% 	const.PropWz   .* dz;
% 
%     [~, minIdx] = min(d);
%     label = nh+ood( lbly(minIdx), lblx(minIdx) );
% 
%     labelsSparse(y, x, viewIdx(1), viewIdx(2)) = label;
%     zmin = z(minIdx);
% 
%     Reproject label into other views based on estimated disparity
%     for v = 1:szLF(3)
%       for u = 1:szLF(4)
% 	xproj = round(min(max(1, x + zmin  * (u - viewIdx(2))), w));
% 	yproj = round(min(max(1, y + zmin  * (v - viewIdx(1))), h));
% 
% 	if ( labelsSparse(yproj, xproj, v, u) == 0)
% 	  labelsSparse( yproj, xproj, v, u ) = label;
% 	end
%       end
%     end
% 
%   end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    end
    

    % Fill in any unlabeled pixels using nearest neighbor assignment
    for i = 1:szLF(3)
      parfor j = 1:szLF(4)
        X(:, :, i, j) = nnFill(X(:, :, i, j), LF(:, :, :, i, j));
      end
    end
%      for i = 1:9
%         for j=1:9
%         imshow(label2rgb(X(:, :,i,j ), colors));
%         title('disagree Map');
%           waitforbuttonpress;
%         end
%     end


    % A labelled light field is ready for consumption.... Enjoy!
    save(['/home/jo/lightfieldsuperpixels/custom/' num2str(szSP) '.mat'], 'X');
  end
