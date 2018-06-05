
%Striatal-sma decisions and 3 factor learning using prediction errors 
%Variable intervals between response and feedback 0, 500, 100ms
%Maddox 2013 BrainAndCognition model

clear all; close all; clc;

tic

numsims=1; %50; 
blocksize = 80;
num_blocks = 5;
conds=3;
numtrials=blocksize*num_blocks; 

%parameters (adjustable) %Glutamate receptor voltage thresholds
nmda=4;
ampa=0.2;

%%learning parameters (adjustable)
ltp=1.5*(10^-9); 
ltd1=0.9*(10^-9); 
ltd2=.005*(10^-9); 

%Time duration & step size
T=3000; tau=1;  n=round(T/tau); %Euler's timescale
startstim = 100; endstim = 800; %stim pres boundaries for FULL feedback
%feedback is 500ms after endstim for con=1  %1700ms allows for smearning

%Postsynaptic MSN parameters 
ddS=200; %SPREAD
delayS=675; %DELAY 675 gave highest for feedback at 500ms compared to 0 and 1000ms 

%initialize variables
correct=zeros(numtrials,conds,numsims);
stimuli=zeros(numtrials,3,conds,numsims);

%load 100x100 stimuli and transform to 25x25 (divide by 4)
stimuli_orig = load('80_100X100_ii_stimDF13.dat'); 
stimuli1 = [stimuli_orig(1:40,1) (stimuli_orig(1:40,2:3)/4)];
stimuli2 = [stimuli_orig(41:80,1) (stimuli_orig(41:80,2:3)/4)];
%m1=mean(stimuli1(:,2:3)); m2=mean(stimuli2(:,2:3));
%c1=cov(stimuli1(:,2:3)); %same as: c2=cov(stimuli2(:,2:3));


%set up stimulus presentation order for each block, each condition and each simulation:
for sim=1:numsims
    for con=1:conds
        stim12 = [stimuli1; stimuli2]; 
        %randomize the 80ABstims for each of the 5 blocks:
        stimuli(1:numtrials,:,con,sim)=[randrows(stim12);randrows(stim12);randrows(stim12);randrows(stim12); randrows(stim12)];
    end
    
end
figure(1);plot2dstim(stim12);title('categorization stimuli');

%Euler's constants:
Cstr=50; vrstr=-80; vtstr=-25; kstr=1;     %MSN -- striatum
astr=0.01; bstr=-20; cstr=-55; dstr=150;   %MSN -- striatum
vpeakstr=40; Estr=100;                     %MSN -- striatum

Cgpi=15; vrgpi=-60; vtgpi=-40; kgpi=0.7;   %GPi parameters (QIF)       
cgpi=-50;  vpeakgpi=35;                    %GPi parameters (QIF)

Cth=1; vrth=-60; vtth=-40; kth=0.7;         %Thalamus parameters (QIF)
cth=-50; vpeakth=35;                      %Thalamus parameters (QIF)

Csma=100; vrsma=-60; vtsma=-40; ksma=0.7;       %SMA, IzRS
asma=0.03; bsma=0; csma=-50; dsma=100;         %SMA, IzRS
vpeaksma=35; Esma=60;                       %SMA, IzRS

%DA firing model constants:
C=100; vr=-60; vt=-40; k=0.7;       %RS -- regular spiking constants
aa=0.03; b=-2; c=-50; d=100;        %RS -- regular spiking constants
vpeak=35; E=71;                     %RS -- regular spiking constants



thresh=5.0;        %SMA output threshold for decision
decision='x';

alpha=0.8; %0.05; %alpha value for input gaussian (used here as a multiplier of distance: bigger alpha thinner gaussian (fine-tuned receptive field), as alpha goes to zero, spread goes to 1
sqwvsize=600;               

onbound=0; %counter for stimulusx=stimulusy - should never happen!

for siml=1:numsims

for con=1:conds

  %Visual cortex to MSN weights
  weights1= ones(25,25); weights2= ones(25,25); %weights1= ones(100,100); weights2= ones(100,100);
  wmax= 1;

  for i=1:25 %100
    for j=1:25 %100
     weights1(i,j)= random('unif',.2,.225);  %Assign weights for vis-cortex
     weights2(i,j)= random('unif',.2,.225);  %to MSN synapses
    end
  end
  
  if con==1; n=3000; end
  if con==2; n=3500; end
  if con==3; n=4000; end
  
  tS=0:1:n-1;
  alphaS=(tS/ddS).*exp((ddS-tS)/ddS);


for trial=1:numtrials
    
    %Begin trial loop
    Igpi1=zeros(1,n); Igpi2=zeros(1,n);      %input/output, lateral inhibition
    Istr1= zeros(1,n); Istr2= zeros(1,n);    %variable instantiation
    Ith1= zeros(1,n); Ith2= zeros(1,n);
    Isma1= zeros(1,n); Isma2= zeros(1,n);
    Ida=zeros(1,n); Ida1=zeros(1,3*n);
    strlat1= zeros(1,n); strlat2= zeros(1,n);
    smalat1= zeros(1,n); smalat2= zeros(1,n);
    smaout1= zeros(1,n); smaout2= zeros(1,n);

    vgpi1=vrgpi*ones(1,n);  vgpi2=vrgpi*ones(1,n);        %V,u instantiation
    vstr1=vrstr*ones(1,n); ustr1=0*vstr1;
    vstr2=vrstr*ones(1,n); ustr2=0*vstr2;
    vth1=vrth*ones(1,n);  vth2=vrth*ones(1,n);
    vsma1=vrsma*ones(1,n); usma1=0*vsma1;
    vsma2=vrsma*ones(1,n); usma2=0*vsma2;
    
    vsmear1 = zeros(1,n); vsmear2 = zeros(1,n);
    

    stimulusx = stimuli(trial,2,con,siml); %x coordinate
    stimulusy = stimuli(trial,3,con,siml); %y coordinate
    
    inputCells= ones(25,25); %inputCells= ones(100,100);
    for i=1:25 %100
        for j=1:25 %100
         dist= sqrt((i-stimulusx)^2 + (j-stimulusy)^2);  
         inputCells(i,j)= exp(-alpha*dist^2);
        end
    end
    vcactivation= inputCells*sqwvsize;

    vcmsnsignalsize1= weights1.*vcactivation;    %Matrices containing signal size of 
    vcmsnsignalsize2= weights2.*vcactivation;    %each vc to striatum synapse


    %Sum inputs to striatal MSNs across stimulus duration 
    for i=startstim:endstim
        alphain1=0; alphain2=0;
     for j=1:25 %100
        for q=1:25 %100
          alphain1= alphain1 + vcmsnsignalsize1(j,q);
          alphain2= alphain2 + vcmsnsignalsize2(j,q);  
        end
     end
     Istr1(i)= alphain1;
     Istr2(i)= alphain2;
    end

%Euler's method loop
for i=1:n-1
    
 %MSN voltage changes    
 vstr1(i+1)=vstr1(i)+tau*(kstr*(vstr1(i)-vrstr)*(vstr1(i)-vtstr)-ustr1(i)+Estr-100*strlat1(i)+Istr1(i)+normrnd(0,20))/Cstr;
 ustr1(i+1)=ustr1(i)+tau*astr*(bstr*(vstr1(i)-vrstr)-ustr1(i));
 
 if vstr1(i+1)>=vpeakstr
        vstr1(i)=vpeakstr;
        vstr1(i+1)=cstr;
        ustr1(i+1)= ustr1(i+1)+dstr;
 end;
 
 
 vstr2(i+1)=vstr2(i)+tau*(kstr*(vstr2(i)-vrstr)*(vstr2(i)-vtstr)-ustr2(i)+Estr-100*strlat2(i)+Istr2(i)+normrnd(0,20))/Cstr;
 ustr2(i+1)=ustr2(i)+tau*astr*(bstr*(vstr2(i)-vrstr)-ustr2(i));
 
 if vstr2(i+1)>=vpeakstr
        vstr2(i)=vpeakstr;
        vstr2(i+1)=cstr;
        ustr2(i+1)= ustr2(i+1)+dstr;
 end;
  
 %Generate alpha fn input to GPi & other MSN when MSN spikes
 if vstr1(i)==vpeakstr 
   for j=i:n
    t= j-i;
    Igpi1(j)= Igpi1(j)+(t*exp((100-t)/100))/100;  
    strlat2(j)= strlat2(j)+(t*exp((100-t)/100))/100; 
   end
 end
 
 
 if vstr2(i)==vpeakstr 
   for j=i:n
    t= j-i;
    Igpi2(j)= Igpi2(j)+(t*exp((100-t)/100))/100;  
    strlat1(j)= strlat1(j)+(t*exp((100-t)/100))/100; 
   end
 end
 
 %GPi voltage changes
 vgpi1(i+1)=vgpi1(i)+tau*(kgpi*(vgpi1(i)-vrgpi)*(vgpi1(i)-vtgpi)+71-.4175*Igpi1(i))/Cgpi;  
 
 if vgpi1(i+1)>=vpeakgpi
        vgpi1(i)=vpeakgpi;
        vgpi1(i+1)=cgpi;
 end;
 
 vgpi2(i+1)=vgpi2(i)+tau*(kgpi*(vgpi2(i)-vrgpi)*(vgpi2(i)-vtgpi)+71-.4175*Igpi2(i))/Cgpi;  
 
 if vgpi2(i+1)>=vpeakgpi
        vgpi2(i)=vpeakgpi;
        vgpi2(i+1)=cgpi;
 end;
 
 %Generate alpha fn input to Thalamus when GPi spikes
  if vgpi1(i)==vpeakgpi
   for j=i:n
    t= j-i;
    Ith1(j)= Ith1(j)+(t*exp((100-t)/100))/100;    
   end
  end
 
  if vgpi2(i)==vpeakgpi
   for j=i:n
    t= j-i;
    Ith2(j)= Ith2(j)+(t*exp((100-t)/100))/100;    
   end
  end
 
 %Thalamus voltage changes
 vth1(i+1)=vth1(i)+tau*(kth*(vth1(i)-vrth)*(vth1(i)-vtth)+71-0.275*Ith1(i))/Cth;
 
 if vth1(i+1)>=vpeakth
        vth1(i)=vpeakth;
        vth1(i+1)=cth;
 end;
 
 vth2(i+1)=vth2(i)+tau*(kth*(vth2(i)-vrth)*(vth2(i)-vtth)+71-0.275*Ith2(i))/Cth;
 
 if vth2(i+1)>=vpeakth
        vth2(i)=vpeakth;
        vth2(i+1)=cth;
 end;
 
 %Generate alpha fn input to SMA when Thalamus spikes
 if vth1(i)==vpeakth
   for j=i:n
    t= j-i;
    Isma1(j)= Isma1(j)+(t*exp((100-t)/100))/100;    
   end
 end
 
 if vth2(i)==vpeakth
   for j=i:n
    t= j-i;
    Isma2(j)= Isma2(j)+(t*exp((100-t)/100))/100;    
   end
 end
 
 
 %SMA voltage changes
 vsma1(i+1)=vsma1(i)+tau*(ksma*(vsma1(i)-vrsma)*(vsma1(i)-vtsma)-usma1(i)+Esma-1.25*smalat1(i)+2*Isma1(i)+normrnd(0,20))/Csma;
 usma1(i+1)=usma1(i)+tau*asma*(bsma*(vsma1(i)-vrsma)-usma1(i));
    if vsma1(i+1)>=vpeaksma
        vsma1(i)=vpeaksma;
        vsma1(i+1)=csma;
        usma1(i+1)= usma1(i+1)+dsma;
    end;
    
 vsma2(i+1)=vsma2(i)+tau*(ksma*(vsma2(i)-vrsma)*(vsma2(i)-vtsma)-usma2(i)+Esma-1.25*smalat2(i)+2*Isma2(i)+normrnd(0,20))/Csma;
 usma2(i+1)=usma2(i)+tau*asma*(bsma*(vsma2(i)-vrsma)-usma2(i));
    if vsma2(i+1)>=vpeaksma
        vsma2(i)=vpeaksma; 
        vsma2(i+1)=csma;
        usma2(i+1)= usma2(i+1)+dsma;
    end;
    
 %SMA lateral inhibition & alpha fn output  
    if vsma1(i)==vpeaksma
     for j=i:n
     t= j-i;
     smalat2(j)= smalat2(j)+(t*exp((100-t)/100))/100; 
     smaout1(j)= smaout1(j)+(t*exp((100-t)/100))/100;
     end
    end

    if vsma2(i)==vpeaksma
     for j=i:n
     t= j-i;
     smalat1(j)= smalat1(j)+(t*exp((100-t)/100))/100; 
     smaout2(j)= smaout2(j)+(t*exp((100-t)/100))/100;
     end
    end
    
%Apply alpha function for MSN voltage: To be used in learning equations

    if vstr1(i)>0
        if i<n-delayS
            efftS=zeros(1,n);
            efftS=[efftS(1:i+delayS),alphaS(1:n-i-delayS)];
            vsmear1=vsmear1+efftS;
        end
    end
    if vstr2(i)>0
        if i<n-delayS
            efftS=zeros(1,n);
            efftS=[efftS(1:i+delayS),alphaS(1:n-i-delayS)];
            vsmear2=vsmear2+efftS;
        end
    end  

        
end %ran through n-1 ms within a trial


%Make decision based on sma output
for i=1:n-1
  if smaout1(i)>=thresh && smaout2(i)<thresh
      decision='a';
      break;
  end
  if smaout2(i)>=thresh && smaout1(i)<thresh
      decision='b';
      break;
  end
  if smaout2(i)>=thresh && smaout1(i)>=thresh
      a= round(rand);
      if a==1
         decision='a';
      else
         decision='b';
      end
      break;
  end
  if i==n-1
      a= round(rand);
      if a==1
         decision='a';
      else
         decision='b';
      end
      break;
  end
end

%Incorporate feedback via dopamine levels
obr=0; 
if trial==1
   pr=0; %0.5; %
end

if (stimulusx<stimulusy && decision=='a')||(stimulusy<stimulusx && decision=='b')
   obr=1; 
   correct(trial,con,siml)=1;
     
end
if (stimulusx>stimulusy && decision=='a')||(stimulusy>stimulusx && decision=='b')
   obr=0; %-1; %
end
%this should never happen bc I'm sampling from a pre-generated ii_stim that
%is not overlapping categories...
if stimulusx==stimulusy
   obr=0; onbound=onbound+1;
end

rpe= obr-pr;

if con==1; fd=0; elseif con==2; fd=500; elseif con==3; fd=1000; end

% % % %Reward prediction and Reward Prediction Error for DA input
DAn=n; %2000ms 1 stimulus 1 feedback
DAin=zeros(DAn,1);

DAin(startstim:startstim+100)=pr; 
DAin(endstim+fd:endstim+fd+100)=rpe; 

% % % if rpe>1
% % %  Dlevel=1;
% % % end
% % % if rpe<=1 && rpe>-.25
% % %  Dlevel= .8*rpe +.2;
% % % end
% % % if rpe<=-.25
% % %  Dlevel=0;
% % % end

%update reward prediction for next trial
pr= pr + 0.075*rpe;

 
% % % % Set up DA firing model
% % % 
vv=vr*ones(1,DAn+700); u=0*vv;

DAstart=zeros(700,1);

DAstart_in = [DAstart; DAin]; %ramp up to baseline takes 674ms, so start 700ms early to stablize baseline activity


Deff=zeros(1,DAn+700);
dd=100; %SPREAD - reduced this from 200: too much smearing
tD=0:1:(DAn+700)-1;
alphaD=(tD/dd).*exp((dd-tD)/dd);
delayD=0; %DELAY

for i=1:(DAn+700)-1
    vv(i+1)=vv(i)+tau*(k*(vv(i)-vr)*(vv(i)-vt)-u(i)+E+500*DAstart_in(i)+normrnd(0,.001))/C; %500*DAstart_in(i)
    u(i+1)=u(i)+tau*aa*(b*(vv(i)-vr)-u(i));
    if vv(i+1)>=vpeak
        vv(i)=vpeak;
        vv(i+1)=c;
        u(i+1)= u(i+1)+d;
        if i<(DAn+700)-delayD
            efft=zeros(1,DAn+700);
            efft=[efft(1:i+delayD),alphaD(1:(DAn+700)-i-delayD)];
            Deff=Deff+efft;
        end
    end
end

    DAeff = Deff(701:DAn+700);
    
    meanDAbase=mean(DAeff((DAn-600):DAn));
    DAeff = DAeff - meanDAbase;
    DAbase = mean(DAeff((DAn-600):DAn));


%learning equations to update the weights
 
temp1= zeros(25,25); temp2= zeros(25,25); %temp1= zeros(100,100); temp2= zeros(100,100);

%update weights for stimuli activated in current trial:
Iksums= zeros(25,25); %Iksums= zeros(100,100);
    
vsum1=sum(vsmear1)/1000;
vsum2=sum(vsmear2)/1000;

%Calculate Product of MSN and DA alpha functions
DAProd1 = sum(DAeff.*vsmear1)/1000;
DAProd2 = sum(DAeff.*vsmear2)/1000;

%CHANGED FROM 10^-4 TO 10^-2 FOR LTP

for i=1:25 %100
     for j=1:25 %100
      Iksums(i,j)=vcactivation(i,j)*(endstim-startstim)*tau;
      
        %3 parts of the equation for weights1 (CAT A)
         if vsum1>nmda && DAProd1>DAbase
           
          temp1(i,j)= temp1(i,j)+ltp*Iksums(i,j)*DAProd1*(wmax-weights1(i,j)); 
         end
         if vsum1>nmda && DAProd1<DAbase

          temp1(i,j)= temp1(i,j)-ltd1*Iksums(i,j)*(DAbase-DAProd1)*(weights1(i,j)); 
         end
         if vsum1<nmda && vsum1>ampa
          temp1(i,j)= temp1(i,j)-ltd2*Iksums(i,j)*(nmda-vsum1)*(vsum1-ampa)*(weights1(i,j));
         end
         %3 parts of the equation for weights2 (CAT B)
         if vsum2>nmda && DAProd2>DAbase
            
          temp2(i,j)= temp2(i,j)+ltp*Iksums(i,j)*DAProd2*(wmax-weights2(i,j));  
         end
         if vsum2>nmda && DAProd2<DAbase

          temp2(i,j)= temp2(i,j)-ltd1*Iksums(i,j)*(DAbase-DAProd2)*(weights2(i,j)); 
         end 
         if vsum2<nmda && vsum2>ampa
          temp2(i,j)= temp2(i,j)-ltd2*Iksums(i,j)*(nmda-vsum2)*(vsum2-ampa)*(weights2(i,j)); 
         end


        %UPDATE weights:  
        weights1(i,j)=weights1(i,j)+temp1(i,j);
        weights2(i,j)=weights2(i,j)+temp2(i,j);
     end
end

 
%FOR VISUALIZING PROGRESS:
%   trialnum_correct_DAbase_DAProds_nmda_vsums = [trial correct(trial,con,siml) DAbase DAProd1 DAProd2 nmda vsum1 vsum2]
if  trial==80  %  ((trial==1)||(trial==20)||(trial==80))  %
    thr=ones(n,1)*thresh; nmd=ones(n,1)*nmda;
    figure(2); subplot(3,3,con); plot(smaout1,'blue');hold on; plot(smaout2,'red');hold on;plot(thr,'black'); legend('SMA unit 1','SMA unit 2', 'Decision threshold');
    subplot(3,3,con+3);plot(vstr1,'blue');hold on;plot(vstr2,'red');hold on;hold on; legend('MSN spikes unit 1','MSN spikes unit 2'); 
    subplot(3,3,con+6);plot(vsmear1,'blue');hold on;plot(vsmear2,'red');hold on;plot(nmd,'black');hold on; plot(DAeff,'green'); legend('MSN unit 1','MSN unit 2', 'NMDA threshold', 'DA'); 
end


end %end of trials loop

%plot weights for this simulation
  figure(3);subplot(2,3,con); surf(weights1);title('weights for category 1');
  subplot(2,3,con+3);surf(weights2);title('weights for category 2');

end %end of conditions loop

end %end of simulations loop


%Calculate % learned over 80 trial blocks
learned= ones(num_blocks,conds);
learnedeach = ones(num_blocks,conds,numsims);
for sim=1:numsims
    for con=1:conds
        for i=1:num_blocks
            sum1=0;
            for j=1:blocksize
                sum1=sum1+correct((i-1)*blocksize+j,con,sim); %correct(j,sim);
            end
            learned(i,con) = learned(i,con) + 100*sum1/blocksize;
            learnedeach(i,con,sim) = learnedeach(i,con,sim) + 100*sum1/blocksize;
        end
    end
end


learned0=learned(:,1)/numsims;    
learned500=learned(:,2)/numsims;
learned1000=learned(:,3)/numsims;

figure(4);plot(learned0,'red'); hold on; plot(learned500,'black'); hold on; plot(learned1000,'blue'); legend('0ms delay','500ms delay','1000ms delay'); title('Learning curves over 5 blocks of 80 trials');

toc;