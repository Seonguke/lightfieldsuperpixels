
colors = rand( 2000, 3);
addpath('./prop', './lines', './seg', './util');%검색폴더 경로 추가
fin = '/home/jo/lightfieldsuperpixels/HCI/stillLife/labels.h5'
labels=load('/home/jo/lightfieldsuperpixels/results/VCLFS/stillLife20.mat').N;
labelsGT = h5read(fin, '/GT_LABELS');
% ll =h5info('/home/jo/바탕화면/new_HCI_DATA/oldhci/blender/papillon/disparity_local.h5');
%%%%%%%%%boundary recall%%%%%%%%%%%%%%%%%
% Chessboard distance range
d = 2; 

[h, w, ~, ~] = size(labels);

[X, Y] = meshgrid(1:w, 1:h);
winX = repmat(-d:d, d * 2 + 1, 1);
winX = winX(:);
winY = repmat([-d:d]', d * 2 + 1, 1);

B_err = zeros(size(labels, 3), size(labels, 4));

for v = 1:size(labels, 3)
  for u = 1:size(labels, 4)
    gt = labelsGT(:, :, v, u); 
    [gMagGT, gdirGT] = imgradient(gt);
    [gMagOurs, gDirOur] = imgradient(labels(:, :, v, u));
    
    % Get indices of pixels with non-zero gradient values (edges)
    mask = gMagGT > 0;
    x = X(mask);
    y = Y(mask);

    % Compute the indices of all edge neighbors
    xall = bsxfun(@plus, x', winX);
    yall = bsxfun(@plus, y', winY);
    xall = max(1, min(w, xall(:)));
    
    yall = max(1, min(h, yall(:)));
    ind = sub2ind([h, w], yall, xall);
    
    % Check if there exist an edge within a distance, in our segmentation
    edgeNeighbors = gMagOurs(ind);
    edgeNeighbors = reshape(edgeNeighbors, (2 * d + 1)^2, []);
    edgeNeighborExist = sum(edgeNeighbors, 1);

    B_err(v, u) = length(find(edgeNeighborExist))/ size(edgeNeighborExist, 2);
  end
end
B_err = mean(mean(B_err));

%%%%%%%%%%%undersegmentation
 [h, w, ~, ~] = size(labels);
  U_err = zeros(size(labels, 3), size(labels, 4));

  for u = 1:size(labels, 4)
    for v = 1:size(labels, 3)
      gt = labelsGT(:, :, v, u);
      label = labels(:,:, v, u);
      uniqueLabelsGT = unique(gt(:));
      UE = 0;

      for i = uniqueLabelsGT(:)'
        % For each unique ground truth labelled region, get the unique superpixel
	% labels that fall within that region
	mask = gt == i;
	overlapLabels = unique(label(mask));
	out = zeros(1, numel(overlapLabels));

	% Calculate the undersegmentation error based on the overlap between
	% the total number of pixels for a superpixel label, and the number of 
	% pixels fo the superpixel label that fall within a single GT region
	for j = 1:numel(overlapLabels)
	  regionNum = sum(sum(label == overlapLabels(j) ));
	  regionGTNum = sum(sum(label(mask) == overlapLabels(j) ));
            
          % the superpixel is within the object segmentation region
          if regionGTNum == regionNum 
            continue
          elseif regionGTNum > regionNum/2 % most of the regions are overlapping
            out(j) = out(j) + regionNum - regionGTNum;
          else 
	    % small overlapping region - the overlapped is regarded as error 
	    % (avoid big panelty for large superpixels)
            out(j) = out(j) + regionGTNum;
          end 
	end
	out = sum(out);
	UE = UE + out/sum(sum(mask));
      end
      U_err(v, u) = UE/length(uniqueLabelsGT);

    end
  end

  U_err = mean(mean(U_err));
%%%%%%%Achievable Accuracy%%%%%%%%%%%%%%

  % Non-labeled pixels cause indexing errors. Deal with them...
  labels(labels == 0) = max(max(max(max(labels)))) + 1;

  errView = zeros(size(labels, 1), size(labels, 2), 3, size(labels, 3), size(labels, 4));
  A_err = zeros(size(labels, 3), size(labels, 4));

  % For each unique label in the superpixel segmentation, find the ground truth
  % label that has maximum overlap with it
  %
  uniqueLabels = unique(labels);
  labelsXgt = zeros( size(uniqueLabels) );

  for i = uniqueLabels(:)'
    maxGtLabelOverlap = 0;

    for u = 1:size(labels, 4)
      for v = 1:size(labels, 3)
        mask = labels(:, :, v, u) == i;
        if ~any(mask)
          continue;
        end

        gt = labelsGT(:, :, v, u);
        [gtLabel, gtLabelOverlap] = mode(gt(mask));

        if gtLabelOverlap > maxGtLabelOverlap
          labelsXgt(i) = gtLabel;
          maxGtLabelOverlap = gtLabelOverlap;
        end

      end
    end
  end

  % Calculate the number of pixels in each view that has the wrong label.
  % Also create a visualization of the error.
  %
  for u = 1:size(labels, 4)
    for v = 1:size(labels, 3)
      l = labels(:, :, v, u);
      gt = labelsGT(:, :, v, u);
      lxgt = labelsXgt(l);

      % Calculate percentage of correct labels in each view
      wrongLabelMask = lxgt ~= gt;
      A_err(v, u) = sum(sum(~wrongLabelMask)) ./ numel(lxgt);

      % Visualize the achievable accuracy for each view  with wrong labels in red
      errRGB = ind2rgb(im2uint8(mat2gray(lxgt)), parula(256));
      errRGB = reshape(errRGB, [], 3);
      errRGB(wrongLabelMask, 1) = 1;
      errRGB(wrongLabelMask, 2) = 0;
      errRGB(wrongLabelMask, 3) = 0;
      errRGB = reshape(errRGB, size(lxgt, 1), size(lxgt, 2), 3);
      errView(:,:,:, v, u) = errRGB;
    end
  end
%  for u = 1:size(labels, 4)
%     for v = 1:size(labels, 3)
%         imshow(errView(:,:,:,v,u));
%         waitforbuttonpress;
%     end
%  end
  A_err = mean(mean(A_err));