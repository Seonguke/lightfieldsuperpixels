lb= load('20.mat');
su= load('epi/papillon/cl_U.mat')
sv= load('epi/papillon/cl_V.mat')
%     for i = 1:9
%         for j=1:9
%         imshow(label2rgb(lb.X(:, :,i,j ), colors));
%         title('disagree Map');
%        
%         waitforbuttonpress;
%         end
%     end
lu= load('epi/papillon/Seg_u.mat');
lv= load('epi/papillon/Seg_v.mat');
colors = rand( 2000, 3);
% imshow( label2rgb(lb.X(:, :,9,9 ), colors) );
X=lb.X;%512x512x9x9 y x v u
Lu=squeeze(su.U(:,:,5,:));
Lv=squeeze(sv.V(:,:,:,5));
for i=1:9
 for i=1:9
 imshow( label2rgb(su.U(:,:, i ), colors) );
waitforbuttonpress;
 end
end
for i=1:9
% for i=1:9
 imshow( label2rgb(Lu(:,:, i ), colors) );
waitforbuttonpress;
end
u=lu.Seg_u;%y,x_u
v=lv.Seg_v;%y,x_v
len=size(Lu,1);
u_err=zeros(len);
v_err=zeros(len);
for i=1:9
% for i=1:9
 imshow( label2rgb(u(:,:, i ), colors) );
waitforbuttonpress;
end
for i=1:9
% for i=1:9
 imshow( label2rgb(v(:,:, i ), colors) );
waitforbuttonpress;
end
% 
for x =1:len %unique label
    uq=unique(v(:,x,:));
    uq;
     for i= 1: length(uq)-1 %x unique label
         mask=zeros(len,9);
         for k=1:9
            for y=1:len
                if v(y,x,k)==i
                   mask(x,k)=1;%find mask
               end
           end
        end
        llv=squeeze(Lv(:,x,:));
         errfind= llv.*mask(:,:);%해당 마스크 곱해줘
         
         
         err_uq=unique(errfind);%거기서 유니크 갑 뽑기~
         if(length(err_uq))>2% 0이랑 1로 쭉 안되어있으면 한줄에 label이 2개 이상 있다는 뜻인거 같다.
             sz=length(err_uq)-1;
             err_f = zeros(sz);%0 제외 해주기~
              for fd= 1:length(err_uq)-1 %err_up+1)
                 err_f(fd)=nnz(errfind==err_uq(fd+1));
              end
             v_max=max(err_f(:,1));
             for fd= 1:length(err_uq) -1 
                 if err_f(fd,1)==v_max;
                     err_f(fd,1)=0;
                 end
             end
             v_err(x)=sum(err_f(:,1));
          end
     end
end
%%%%%%%%%%%%%%%% U_epi error %%%%%%%%%%%%%%%%%%%%
for y =1:len %unique label
    uq=unique(u(y,:,:));
    uq;
    for i= 1: length(uq)-1 %x unique label
        mask=zeros(9,len);
        for k=1:9
           for x=1:len
               if u(y,x,k)==i
                   mask(k,x)=1;%find mask
               end
           end
        end
        
        EPIuc = permute(Lu, [1 3 2] );%    9   512     3   512(u,x,color,y)
        llu=squeeze(EPIuc(y,:,:));
        errfind= llu.*mask(:,:);%해당 마스크 곱해줘 
        
        err_uq=unique(errfind);%거기서 유니크 갑 뽑기~
        if(length(err_uq))>2% 0이랑 1로 쭉 안되어있으면 한줄에 label이 2개 이상 있다는 뜻인거 같다.
            sz=length(err_uq)-1;
            err_f = zeros(sz);%0 제외 해주기~
             for fd= 1:sz %err_up+1)
                err_f(fd)=nnz(errfind==err_uq(fd+1));
             end
              u_max=max(err_f(:,1));
              for fd= 1:length(err_uq) -1
                  if err_f(fd,1)==u_max;
                      err_f(fd,1)=0;
                  end
              end
             
            u_err(y)=sum(err_f(:,1));   
        end
    end
end
vr=sum(v_err(:,1));
ur=sum(u_err(:,1));
t_err=vr+ur;

%%%%%%%% look at epi%%%%%%%%%%%%%%%
% epi_gen=zeros(9,512);
%     for j=1:512
%         for i=1:9
%             epi_gen(i,:)=u(j,:,i)
%         end
%         epi= label2rgb(epi_gen, colors)
%         rsz=imresize(epi,5)
%         imshow(rsz)
%         waitforbuttonpress;
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%