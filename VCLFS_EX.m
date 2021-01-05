

%%
%% Generate *View Consistent Light Field Superpixels* for the 
%% input light field  using the parameter settings in param
%%

addpath('./prop', './lines', './seg', './util');%검색폴더 경로 추가
D = '/home/jo/lightfieldsuperpixels/cotton.h5'
param = parameters;
cluster = parcluster('local');
cluster.NumWorkers = param.nWorkers;%병렬처리
saveProfile(cluster);
LF = HCIloadLF( D, 'X');

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
  LFvc = permute(LFvc, [2 1 3 4]);
  %size(LFvc)
  for i = 1:size(LFvc, 4) 
    LFvc(:, :, :, i) = imgaussfilt( squeeze(LFvc(:, :, :, i)), 0.85);
  end
  EPIvc = permute(LFvc, [4 2 3 1]);%9   512     3   512
%size(EPIvc)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  % EDGE DETECTION & LINE FITTING  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Lv = {}; Lu = {};
  segMinWidth = round(const.SegMinWidthMultiplier * szLF(3));%2 ??

  for i = param.linesPyramidLevels - 1:-1:0 %% 1, 0
    % Scale-up results from lower pyramid level 2
    LvScaled = cellfun(@(l) l .* (2), repelem(Lv, 2, 1), 'UniformOutput', false);%.*배열연산
    LuScaled = cellfun(@(l) l .* (2), repelem(Lu, 2, 1), 'UniformOutput', false);%cell형태로 값을 반환
    % Fit lines at current level...
    [Lv, Ev] = lines(LFvc, i, param);
    [Lu, Eu] = lines(LFuc, i, param);   % ... and merge with lower pyramid level results	
    Lv = mergeLines(Lv, LvScaled, true, segMinWidth); 
    Lu = mergeLines(Lu, LuScaled, true, segMinWidth);
  end
  %L= outlier regjection  E= Confidence
  
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
  %%%%%%%%%%%%%%%%%%%ex%%%%%%%%%%%%%%%%%%%%%%%%%
  disMap=dMapv
  szEPI = [size(EPIvc, 1) size(EPIvc, 2) size(EPIvc, 4)];
  features = cell(size(Mv));

   for i = 1:szEPI(3)
     m = Mv{i};
     l = Lv{i};

     % Split matched segments that are longer than a threshold
     [m, l] = matchSplit(m, l, disMap(i, :), szEPI);

     % Get the average LAB color of the EPI segment bounded by the matched lines
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MATCH FEATURESLAB%%%%%%%%%%%%%%% 
     %[featuresLAB, featuresIdx] = matchFeaturesLAB(m, l, EPIuc(:, :, :, i));
     
     % Sort the matches/segments by the slope of its bounding lines
    [~, o] = sort(min(l(m(:, 1), 2) - l(m(:, 1), 1), l(m(:, 2), 2) - l(m(:, 2), 1)));

    % Create a mask for each match. 
    % Having the masks of all matches is useful when determining the color of occluded regions.
    %
    labels = zeros(size(EPIvc(:, :, :, i), 1), size(EPIvc(:, :, :, i), 2));
    sz = [size(EPIvc(:, :, :, i), 1) size(EPIvc(:, :, :, i), 2)];
    for j = 1:length(o) 
        mask = rightMask(l(m(o(j), 1), :), sz) & leftMask(l(m(o(j), 2), :), sz);
        labels(mask) = j;
    end
    
    featuresLAB = zeros( size(m, 1), 3);

    % Some segments may be completely occluded. We identify these...
    featuresIdx = zeros( size(m, 1), 1, 'logical');
    
    for j = 1:size(m, 1)
     bw = labels == j;%label이 j인곳 찾아
     s = sum(sum(bw));
        if s ~= 0 
        featuresIdx( o(j) ) = 1;
        featuresLAB(j, :) = permute(sum(sum(EPIvc(:, :, :, i) .* bw)) ./ s, [1 3 2]);
        end
    end
    
    % Put the features back in original order
    [~, p] = sort(o);
    featuresLAB = featuresLAB(p, :);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
     % Discard matched segments that are entirely occludedww
     m = m(featuresIdx, :);
     featuresLAB = featuresLAB(featuresIdx, :);

     % The final set of features has size (n x 8) and is made up of 
     % the top and bottom intercepts of the two bounding linews (4 features), 
     % the EPI index (1 feature), and the LAB color (3 features)
     features{i} = [ matches2parallelograms(m, l), repelem(i, size(featuresLAB, 1), 1), featuresLAB];
     Lu{i} = l;
   end

disp('hi')
disp('hi')
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [Su, Lu] = matches2Features(Mu, Lu, dMapv', EPIuc);
  [Sv, Lv] = matches2Features(Mv, Lv, dMapv, EPIvc);%Sv= 4point segment set , epiindex, Labcolor n*8
 
  for szSP = param.szSuperpixels

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % SPATIO-ANGULAR SEGMENTATION VIA CLUSTERING %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % We use the term "segment" for the features of a pair of matched lines
    [labelsu, labelsv] = segs2clusters(Sv, Su, szLF, szSP, param.wxy, param.wz, param.wlab); 

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
    parfor i = [1:cviewIdx cviewIdx + 1:szLF(4)]
      viewOffset = cviewIdx - i;
      U(:, :, i, :) = permute( segs2labels( segsReproject(Su, -viewOffset, szLF),...
					   szEPIu), [3 2 1]);
      V(:, :, :, i) = permute( permute( segs2labels( segsReproject(Sv, -viewOffset, szLF), ...
						    szEPIv), [3 2 1]), [2 1 3]);
    end

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

    % Disparity estimated from the edge line slopes of the angular segments is
    % used in the propagation of labels
    D = ones(szLF);
    D(:, :, :, cviewIdx) = permute( segs2disparity(Su, szEPIu), [3 2 1] );
    D(:, :, cviewIdx, :) = permute(permute( segs2disparity(Sv, szEPIv), ...
				   [3 2 1] ), [2 1 3]);

    % Light field view indices starting from the center outward, on the central
    % crosshair of views:
    %           :
    %           6 
    %           2 
    %    ...7 3 0 1 5 ...
    %           4
    %           8
    %           :
    %
    theta = [0:pi/2:2*pi*(cviewIdx - 1) - pi/2];
    v = round([cviewIdx cviewIdx + sin(theta) .* repelem([1:cviewIdx - 1], 1, 4)]);
    u = round([cviewIdx cviewIdx + cos(theta) .* repelem([1:cviewIdx - 1], 1, 4)]);

    % Propagate the labels within each "crosshair" view and then project into
    % *all* other views
    for i = 1:length(v)
      X = propagate( X, LF, D(:, :, v(i), u(i)), [v(i) u(i)], szLF);
    end

    % Fill in any unlabeled pixels using nearest neighbor assignment
    for i = 1:szLF(3)
      parfor j = 1:szLF(4)
        X(:, :, i, j) = nnFill(X(:, :, i, j), LF(:, :, :, i, j));
      end
    end

    % A labelled light field is ready for consumption.... Enjoy!
    save([fout num2str(szSP) '.mat'], 'X');
  end
