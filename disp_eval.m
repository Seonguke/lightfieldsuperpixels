  disparityRange = [-2 2];
colors = rand( 2000, 3);
addpath('./prop', './lines', './seg', './util');%검색폴더 경로 추가

labels=load('/home/jo/lightfieldsuperpixels/table_1_5/20.mat').X;
disparityGT=load('/home/jo/lightfieldsuperpixels/table_1_5/table_disp_gt.mat').X;
% for i=1:9
%     for j=1:9
% 
%    
%         imshow(disparityGT(:,:,i,j),disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet); 
%         colorbar;
%      waitforbuttonpress;
%     end
% end
% for i=1:9
%     for j=1:9
% 
%    imshow(label2rgb(labels(:, :,i,j ), colors));
%         title('U');
%          waitforbuttonpress;
%     end
% end

  [h, w, nv, nu] = size(labels);

  % row, column subscripts
  [x, y] = meshgrid(1:w, 1:h);

  % The projected (x, y) centroid of each superpixel (out of a total of (max(labels(:))) 
  % unique superpixels), in each of the nv * nu views.
  pc = ones(max(labels(:)), 2, nv, nu) * inf;

  for v = 1:nv
    for u = 1:nu
      labeluv = labels(:, : ,v, u);

      % Project each pixel into the center view based on the disparity of this view
      px = x + disparityGT(:, :,  v, u);
      py = y + disparityGT(:, :,  v, u); % subtract here since matlab y goes down

      % calculate the projected superpixel center as the mean of the projected
      % x and y values of that superpixel in the center view
      for i = unique(labeluv(:))'
	mask = labeluv == i;
	if any(mask(:))
          pc(i, :, v, u) = [mean(px(mask)), mean(py(mask))];
	end
      end
    end
  end

  % Calculate the difference from the superpixel centroid in the center view
  pc = pc - pc(:,:, ceil(nv/2), ceil(nu/2));
  S_err  = squeeze(sqrt( sum(pc .^2, 2) ));
  S_err = S_err(:);

  % discard superpixels that are entirely occluded in at least one view
  S_err(isnan(S_err)) = [];
  S_err(isinf(S_err)) = [];
  S_err = mean(S_err);
 [h, w, nv, nu] = size(labels);

  % row, column subscripts
  [x, y] = meshgrid(1:w, 1:h);
  idx = [1:h * w]';

  result = zeros(h, w, nv, nu);

  for v = 1:nv
    for u = 1:nu
      
      % Project each pixel into the center view based on the disparity of this view
      px = round(x + disparityGT(:, :, v, u));
      py = round(y + disparityGT(:, :, v, u)); % subtract here since matlab y goes down

      pxOverflowIdx = px < 1 | px > w;
      pyOverflowIdx = py < 1 | py > h;
      px(pxOverflowIdx) = w; % temporarily, avoid out of range sub2ind transfer
      py(pyOverflowIdx) = h; 
      idxProj = sub2ind( [h, w], py, px);
      idxProj = idxProj(:);

      % Identify pixel indices in the center view which have more than one pixel
      % projecting onto them
      uniqueIdx = unique(idxProj);
      hg = hist(idxProj, uniqueIdx);
      idxProjDup = uniqueIdx(hg > 1);

      % Discard all pixels from the duplicate list
      I = ~(ismember(idxProj, idxProjDup));

      % Discard all out-of-range pixels 
      I(pxOverflowIdx) = false;
      I(pyOverflowIdx) = false;
    
      % Find the the foreground pixel in the duplicate list
      % These are added back to the list
      disparityh = disparityGT(:, :,  v, u);
      idxProjOcc = zeros(length(idxProjDup), 1);
      count = 1;
      for idup = idxProjDup(:)'
        subxy = find(idxProj == idup);
        % foregroud pixel -> min depth -> min disparity
        [~, i] = min(abs(disparityh(subxy)));
        idxProjOcc(count) = subxy(i);
        count = count +1;
      end
      % add back the foreground pixel from the duplicate list
      I(idxProjOcc) = true;

      l = labels(:, :, v, u);
      R = zeros(h, w);
      R(idxProj(I)) = l(idx(I));
      result(:, :, v, u) = R;
    end
  end

  result = reshape(result, h, w, []);
  L_err = zeros(h, w);
  centerView = labels(:, :, ceil(nv/2), ceil(nu/2));

  % Calculate the number of differing labels per pixel
  for i = 1:size(result, 3)
    for j = 1:w
      parfor k = 1:h
        % Consider a 3x3 window around the pixel to account for rounding L_errors
        % when reprojecting
        win = centerView( min(h, max(1, [k-1:k+1])), min(w, max(1, [j-1:j+1])) );
        win = win(:);
        if( ~sum(win == result(k, j, i)) )
	  L_err(k, j) = L_err(k, j) + 1;
        end
      end
    end
  end
 
  L_err = mean(L_err(:)); 
  L_err = L_err + 1; % The pixel has at least one label even if none differ

  
  
  