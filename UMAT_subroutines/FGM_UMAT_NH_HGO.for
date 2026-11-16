      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,
     2 TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,MATERL,NDI,NSHR,NTENS,
     3 NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT,
     4 DFGRD0,DFGRD1,NOEL,NPT,KSLAY,KSPT,KSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      LOGICAL FIRST
      SAVE FIRST
      DATA FIRST /.TRUE./
      CHARACTER*8 MATERL
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2 STRAN(NTENS),DSTRAN(NTENS),DFGRD0(3,3),DFGRD1(3,3),
     3 TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3)
C

C    LOCAL ARRAYS
C ----------------------------------------------------------------
C    BBAR   - DEVIATORIC LEFT CAUCHY-GREEN TENSOR
C    DISTGR - DEVIATORIC DEFORMATION GRADIENT (DISTORTION TENSOR)
C ----------------------------------------------------------------
C
      DIMENSION BBAR(6),DISTGR(3,3),DISTGR_VAL(3,3),BVAL(6),BVAL_SQ(6)
      REAL*8 qdiadq(6)
      REAL*8 IA, JJ, JFTHREE, JTTHREE, JSTHREE, JINV, J_4_THREE
      REAL*8 q1, q2, q3, HGO_k1, HGO_k2, C10, D1_inv_NH, D1_inv_HGO, L 
      REAL*8 TW, FO, FI, SE, alpha, beta
C
      PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0, FOUR=4.D0)
      PARAMETER(NINE=9.D0, EIGHT = 8.D0)
C ----------------------------------------------------------------


      X = COORDS(1)
	  Y = COORDS(2)
	  Z = COORDS(3)
      L = PROPS(1)   ! length scale (model-dependent)

C    JACOBIAN AND DISTORTION TENSOR
	 
      DET=DFGRD1(1, 1)*DFGRD1(2, 2)*DFGRD1(3, 3)
     1   -DFGRD1(1, 2)*DFGRD1(2, 1)*DFGRD1(3, 3)
      IF(NSHR.EQ.3) THEN
        DET=DET+DFGRD1(1, 2)*DFGRD1(2, 3)*DFGRD1(3, 1)
     1         +DFGRD1(1, 3)*DFGRD1(3, 2)*DFGRD1(2, 1)
     2         -DFGRD1(1, 3)*DFGRD1(3,1)*DFGRD1(2, 2)
     3         -DFGRD1(2, 3)*DFGRD1(3, 2)*DFGRD1(1, 1)
      END IF
      SCALE=DET**(-ONE/THREE)
      DO K_1=1, 3
        DO K_2=1, 3
          DISTGR(K_2, K_1)=SCALE*DFGRD1(K_2, K_1)
        END DO
      END DO
	  
      DO K_1=1, 3
        DO K_2=1, 3
          DISTGR_VAL(K_2, K_1)= DFGRD1(K_2, K_1)
        END DO
      END DO	  
	  
	   
      C10_MAT = 4.0D0
      C10_CORE = 2.0D0
	  
      D1_inv_NH_MAT = 40.0D0
      D1_inv_NH_CORE = 20.0D0
	  
	  HGO_k1_MAT = 0.0D0
      HGO_k1_CORE = 2.0D0
	  
	  PI = 4.0D0*ATAN(1.0D0)	  
      alpha = 45.0D0*PI/180.0D0
      beta  = ACOS(1.0D0/SQRT(3.0D0))

      IF (KINC .EQ. 1 .AND. KSTEP .EQ. 1) THEN

c		  3D SETTINGS
		  R0 = 18.0D0
		  SIG = 1.0D0
		  R = SQRT((X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2)
		  STATEV(1)= C10_MAT + (C10_CORE
     1		- C10_MAT) * 0.50D0 * (1.0D0-TANH((R-R0)/SIG))
	 
		  STATEV(2)= D1_inv_NH_MAT + (D1_inv_NH_CORE
     1		- D1_inv_NH_MAT) * 0.50D0 * (1.0D0-TANH((R-R0)/SIG))
	 
		  STATEV(3)= HGO_k1_MAT + (HGO_k1_CORE
     1		- HGO_k1_MAT) * 0.50D0 * (1.0D0-TANH((R-R0)/SIG))
	 
     
      END IF
	  

      C10 = STATEV(1)
      D1_inv_NH  = STATEV(2)
	  HGO_k1 = STATEV(3)
      D1_inv_HGO = 0.0D0
	  HGO_k2 = 1.0D0	 


C    CALCULATE LEFT CAUCHY-GREEN TENSOR
C
      BBAR(1)=DISTGR(1, 1)**2+DISTGR(1, 2)**2+DISTGR(1, 3)**2
      BBAR(2)=DISTGR(2, 1)**2+DISTGR(2, 2)**2+DISTGR(2, 3)**2
      BBAR(3)=DISTGR(3, 3)**2+DISTGR(3, 1)**2+DISTGR(3, 2)**2
      BBAR(4)=DISTGR(1, 1)*DISTGR(2, 1)+DISTGR(1, 2)*DISTGR(2, 2)
     1       +DISTGR(1, 3)*DISTGR(2, 3)
      BBAR(5)=DISTGR(1, 1)*DISTGR(3, 1)+DISTGR(1, 2)*DISTGR(3, 2)
     1         +DISTGR(1, 3)*DISTGR(3, 3)
      BBAR(6)=DISTGR(2, 1)*DISTGR(3, 1)+DISTGR(2, 2)*DISTGR(3, 2)
     1         +DISTGR(2, 3)*DISTGR(3, 3)
  

      BVAL(1)=DISTGR_VAL(1, 1)**2+DISTGR_VAL(1, 2)**2 
     1 + DISTGR_VAL(1, 3)**2
      BVAL(2)=DISTGR_VAL(2, 1)**2+DISTGR_VAL(2, 2)**2
     1 + DISTGR_VAL(2, 3)**2
      BVAL(3)=DISTGR_VAL(3, 3)**2+DISTGR_VAL(3, 1)**2
     1 + DISTGR_VAL(3, 2)**2
      BVAL(4)=DISTGR_VAL(1, 1)*DISTGR_VAL(2, 1)
     1 + DISTGR_VAL(1, 2)*DISTGR_VAL(2, 2) 
     2 + DISTGR_VAL(1, 3)*DISTGR_VAL(2, 3)
      BVAL(5)=DISTGR_VAL(1, 1)*DISTGR_VAL(3, 1)
     1 +DISTGR_VAL(1, 2)*DISTGR_VAL(3, 2)
     2 +DISTGR_VAL(1, 3)*DISTGR_VAL(3, 3)
      BVAL(6)=DISTGR_VAL(2, 1)*DISTGR_VAL(3, 1)
     1 +DISTGR_VAL(2, 2)*DISTGR_VAL(3, 2)
     2 +DISTGR_VAL(2, 3)*DISTGR_VAL(3, 3)


C   Calculate BVAL^2
      BVAL_SQ(1)=BVAL(1)**2+ BVAL(4)**2+ BVAL(5)**2
      BVAL_SQ(2)=BVAL(2)**2+ BVAL(4)**2+ BVAL(6)**2
      BVAL_SQ(3)=BVAL(3)**2+ BVAL(5)**2+ BVAL(6)**2
      BVAL_SQ(4)=BVAL(1)*BVAL(4)+ BVAL(2)*BVAL(4)+ BVAL(5)*BVAL(6)
      BVAL_SQ(5)=BVAL(1)*BVAL(5)+ BVAL(3)*BVAL(5)+ BVAL(4)*BVAL(6)  
      BVAL_SQ(6)=BVAL(2)*BVAL(6)+ BVAL(3)*BVAL(6)+ BVAL(4)*BVAL(5)
    	  
C
C    CALCULATE THE STRESS
C
      TRBBAR=(BBAR(1)+BBAR(2)+BBAR(3))/THREE
	  TRBVAL_SQ = BVAL_SQ(1) + BVAL_SQ(2) + BVAL_SQ(3)
	  TRBVAL = BVAL(1)+BVAL(2)+BVAL(3)
	  
C    CALCULATING THE NECESSARY TERMS	
      JFTHREE =  DET**(-5.D0/3.D0)
	  JTTHREE =  DET**(-2.D0/3.D0)
	  JSTHREE =  DET**(-7.D0/3.D0)
	  J_4_THREE = DET**(-4.D0/3.D0)
	  JINV = DET**(-1.D0)  


c	  3D Cube Body Digonal Fiber Orientation	
      a_1 = SIN(beta)*COS(alpha)
      a_2 = SIN(beta)*SIN(alpha)
      a_3 = COS(beta)

      q1 = DFGRD1(1,1)*a_1 + DFGRD1(1,2)*a_2 + DFGRD1(1,3)*a_3
      q2 = DFGRD1(2,1)*a_1 + DFGRD1(2,2)*a_2 + DFGRD1(2,3)*a_3
      q3 = DFGRD1(3,1)*a_1 + DFGRD1(3,2)*a_2 + DFGRD1(3,3)*a_3
	  
	  qdiadq(1) = q1**2
	  qdiadq(2) = q2**2
	  qdiadq(3) = q3**2
	  qdiadq(4) = q1*q2
	  qdiadq(5) = q1*q3
	  qdiadq(6) = q2*q3	  

	  
      IA = q1*q1 + q2*q2 + q3*q3
      EX =  EXP(HGO_k2*(JTTHREE*IA - 1.D0)**2)
      EG=TWO*C10/DET
      EK=TWO*D1_inv_NH*(TWO*DET-ONE)
      PR=TWO*D1_inv_NH*(DET-ONE)

	  
      DO V_1=1,NDI
        STRESS(V_1)= 2.D0*JFTHREE*(BVAL(V_1)-TRBVAL/3.D0)*C10   
     1 + (DET -JINV)*D1_inv_HGO 
     2 + (HGO_k1/2.D0)*EX*((-4.D0/3.D0)*IA*(IA*JSTHREE - JFTHREE)) 
     3 + (HGO_k1/2.D0)*EX*4.D0*qdiadq(V_1)*(IA*JSTHREE - JFTHREE)
     4 + PR
	 
      END DO
      DO V_1=NDI+1,NDI+NSHR
        STRESS(V_1)= 2.D0*JFTHREE*(BVAL(V_1))*C10 
     1 + (HGO_k1/2.D0)*EX*4.D0*qdiadq(V_1)*(IA*JSTHREE - JFTHREE)
      END DO
C
C    CALCULATE THE STIFFNESS
C
      TW = 2.D0/3.D0
	  FO = 4.D0/3.D0
	  FI = 5.D0/3.D0
	  SE = 7.D0/3.D0

      EG23=EG*TWO/THREE
	  JJ = DET  
	  
      DDSDDE(1, 1)= EG23*(BBAR(1)+TRBBAR)
     1 + 4*EX*(IA/JJ**(SE) 
     2 - JJ**(-(FI)))*HGO_k1*q1**2 
     3 + (EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     4 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-2*IA)/3.D0 + 2*q1**2)*((-4*IA)/3. 
     5 + 4*q1**2))/JJ**(FI) 
     6 + 2*EX*HGO_k1*q1**2*((-4*IA)/(3.*JJ**(SE)) 
     7 + 2/(3.*JJ**(FI)) + (2*q1**2)/JJ**(SE)) 
     8 - (2*EX*HGO_k1*((-4*IA**2)/(3.D0*JJ**(SE)) 
     9 + (2*IA)/(3.*JJ**(FI)) 
     A + (4*IA*q1**2)/JJ**(SE) 
     B - (2*q1**2)/JJ**(FI)))/3.D0 + 2*JJ*D1_inv_HGO	  
     D +  EK
	  
      DDSDDE(2, 2)= EG23*(BBAR(2)+TRBBAR)
     1 + 4*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q2**2+ (EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-2*IA)/3. 
     3 + 2*q2**2)*((-4*IA)/3. 
     4 + 4*q2**2))/JJ**(FI) + 2*EX*HGO_k1*q2**2*((-4*IA)/(3.*JJ**(SE)) 
     5 + 2/(3.*JJ**(FI)) 
     6 + (2*q2**2)/JJ**(SE)) - (2*EX*HGO_k1*((-4*IA**2)/(3.*JJ**(SE)) 
     7 + (2*IA)/(3.*JJ**(FI)) + (4*IA*q2**2)/JJ**(SE) 
     8 - (2*q2**2)/JJ**(FI)))/3. + 2*JJ*D1_inv_HGO	  
     D +  EK
	 
      DDSDDE(3, 3)= EG23*(BBAR(3)+TRBBAR)
     1 + 4*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q3**2+ (EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-2*IA)/3. 
     3 + 2*q3**2)*((-4*IA)/3. + 4*q3**2))/JJ**(FI) 
     4 + 2*EX*HGO_k1*q3**2*((-4*IA)/(3.*JJ**(SE)) + 2/(3.*JJ**(FI)) 
     5 + (2*q3**2)/JJ**(SE)) - (2*EX*HGO_k1*((-4*IA**2)/(3.*JJ**(SE)) 
     6 + (2*IA)/(3.*JJ**(FI)) + (4*IA*q3**2)/JJ**(SE) 
     7 - (2*q3**2)/JJ**(FI)))/3. + 2*JJ*D1_inv_HGO	  
     D +  EK
	 
      DDSDDE(1, 2)=-EG23*(BBAR(1)+BBAR(2)-TRBBAR)
     1 +(EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     2 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-4*IA)/3. 
     3 + 4*q1**2)*((-2*IA)/3. + 2*q2**2))/JJ**(FI) 
     4 + 2*EX*HGO_k1*q1**2*((-4*IA)/(3.*JJ**(SE)) + 2/(3.*JJ**(FI)) 
     5 + (2*q2**2)/JJ**(SE)) - (2*EX*HGO_k1*((-4*IA**2)/(3.*JJ**(SE)) 
     6 + (2*IA)/(3.*JJ**(FI)) + (4*IA*q2**2)/JJ**(SE) 
     7 - (2*q2**2)/JJ**(FI)))/3. + 2*JJ*D1_inv_HGO	  
     D +  EK
	 
      DDSDDE(1, 3)=-EG23*(BBAR(1)+BBAR(3)-TRBBAR)
     1 + (EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     2 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-4*IA)/3. 
     3 + 4*q1**2)*((-2*IA)/3. + 2*q3**2))/JJ**(FI) 
     4 + 2*EX*HGO_k1*q1**2*((-4*IA)/(3.*JJ**(SE)) + 2/(3.*JJ**(FI)) 
     5 + (2*q3**2)/JJ**(SE)) - (2*EX*HGO_k1*((-4*IA**2)/(3.*JJ**(SE)) 
     6 + (2*IA)/(3.*JJ**(FI)) + (4*IA*q3**2)/JJ**(SE) 
     7 - (2*q3**2)/JJ**(FI)))/3. + 2*JJ*D1_inv_HGO	  
     D +  EK
	 
      DDSDDE(2, 3)=-EG23*(BBAR(2)+BBAR(3)-TRBBAR)
     1 + (EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     2 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-4*IA)/3. + 4*q2**2)*((-2*IA)/3. 
     3 + 2*q3**2))/JJ**(FI) + 2*EX*HGO_k1*q2**2*((-4*IA)/(3.*JJ**(SE)) 
     4 + 2/(3.*JJ**(FI)) + (2*q3**2)/JJ**(SE)) 
     5 - (2*EX*HGO_k1*((-4*IA**2)/(3.*JJ**(SE)) 
     6 + (2*IA)/(3.*JJ**(FI)) + (4*IA*q3**2)/JJ**(SE) 
     7 - (2*q3**2)/JJ**(FI)))/3. + 2*JJ*D1_inv_HGO	  
     D +  EK
	 
      DDSDDE(1, 4)= EG23*BBAR(4)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q2 
     2 + (4*EX*HGO_k1*q1**3*q2)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     3 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*((-4*IA)/3. 
     4 + 4*q1**2)*q2)/JJ**(FI) - (2*EX*HGO_k1*((4*IA*q1*q2)/JJ**(SE) 
     5 - (2*q1*q2)/JJ**(FI)))/3.

      DDSDDE(2, 4)= EG23*BBAR(4)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q2 
     2 + (4*EX*HGO_k1*q1*q2**3)/JJ**(SE) 
     3 - (2*EX*HGO_k1*((4*IA*q1*q2)/JJ**(SE) 
     4 - (2*q1*q2)/JJ**(FI)))/3. + (2*EX*(IA/JJ**(FO) 
     5 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*q2*((-4*IA)/3. 
     6 + 4*q2**2))/JJ**(FI)
	  
      DDSDDE(3, 4)=-EG23*BBAR(4)
     1 + (-2*EX*HGO_k1*((4*IA*q1*q2)/JJ**(SE) - (2*q1*q2)/JJ**(FI)))/3. 
     2 + (4*EX*HGO_k1*q1*q2*q3**2)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     3 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*q2*((-4*IA)/3. 
     4 + 4*q3**2))/JJ**(FI)
	 	  
      DDSDDE(4, 4)= EG*(BBAR(1)+BBAR(2))/TWO
     1 + (4*EX*HGO_k1*q1**2*q2**2)/JJ**(SE) + (8*EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 
     3 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1**2*q2**2)/JJ**(FI)
     4 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*(q1**2/2. + q2**2/2.)
	 
	  
      IF(NSHR.EQ.3) THEN
        DDSDDE(1, 5)= EG23*BBAR(5)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q3 
     2 + (4*EX*HGO_k1*q1**3*q3)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     3 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*((-4*IA)/3. 
     4 + 4*q1**2)*q3)/JJ**(FI) - (2*EX*HGO_k1*((4*IA*q1*q3)/JJ**(SE) 
     5 - (2*q1*q3)/JJ**(FI)))/3. 
		
		
        DDSDDE(2, 5)=-EG23*BBAR(5)
     1 + (4*EX*HGO_k1*q1*q2**2*q3)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*((-4*IA)/3. 
     3 + 4*q2**2)*q3)/JJ**(FI) - (2*EX*HGO_k1*((4*IA*q1*q3)/JJ**(SE) 
     4 - (2*q1*q3)/JJ**(FI)))/3.	
	
        DDSDDE(3, 5)= EG23*BBAR(5)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q3 
     2 + (4*EX*HGO_k1*q1*q3**3)/JJ**(SE) 
     3 - (2*EX*HGO_k1*((4*IA*q1*q3)/JJ**(SE) 
     4 - (2*q1*q3)/JJ**(FI)))/3. + (2*EX*(IA/JJ**(FO) 
     5 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*q3*((-4*IA)/3. 
     6 + 4*q3**2))/JJ**(FI)	
		
        DDSDDE(1, 6)=-EG23*BBAR(6)
     1 +(4*EX*HGO_k1*q1**2*q2*q3)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*((-4*IA)/3. 
     3 + 4*q1**2)*q2*q3)/JJ**(FI) - (2*EX*HGO_k1*((4*IA*q2*q3)/JJ**(SE) 
     4 - (2*q2*q3)/JJ**(FI)))/3.	
		
        DDSDDE(2, 6)= EG23*BBAR(6)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q2*q3 
     2 + (4*EX*HGO_k1*q2**3*q3)/JJ**(SE) + (2*EX*(IA/JJ**(FO) 
     3 - JJ**(-(TW)))*(-1 + IA/JJ**(TW))*HGO_k1*HGO_k2*q2*((-4*IA)/3. 
     4 + 4*q2**2)*q3)/JJ**(FI) - (2*EX*HGO_k1*((4*IA*q2*q3)/JJ**(SE) 
     5 - (2*q2*q3)/JJ**(FI)))/3.	
	
        DDSDDE(3, 6)= EG23*BBAR(6)/TWO
     1 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q2*q3 
     2 + (4*EX*HGO_k1*q2*q3**3)/JJ**(SE) 
     3 - (2*EX*HGO_k1*((4*IA*q2*q3)/JJ**(SE) - (2*q2*q3)/JJ**(FI)))/3. 
     4 + (2*EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     5 + IA/JJ**(TW))*HGO_k1*HGO_k2*q2*q3*((-4*IA)/3.+4*q3**2))/JJ**(FI)		
	
        DDSDDE(5, 5)= EG*(BBAR(1)+BBAR(3))/TWO
     1 + (4*EX*HGO_k1*q1**2*q3**2)/JJ**(SE) + (8*EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 
     3 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1**2*q3**2)/JJ**(FI) 
     4 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*(q1**2/2. + q3**2/2.) 
			
        DDSDDE(6, 6)= EG*(BBAR(2)+BBAR(3))/TWO
     1 + (4*EX*HGO_k1*q2**2*q3**2)/JJ**(SE) + (8*EX*(IA/JJ**(FO) 
     2 - JJ**(-(TW)))*(-1 
     3 + IA/JJ**(TW))*HGO_k1*HGO_k2*q2**2*q3**2)/JJ**(FI) 
     4 + 2*EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*(q2**2/2. + q3**2/2.)
		
        DDSDDE(4,5)= EG*BBAR(6)/TWO
     1 + EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q2*q3 
     2 + (4*EX*HGO_k1*q1**2*q2*q3)/JJ**(SE) 
     3 + (8*EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     4 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1**2*q2*q3)/JJ**(FI)
		
        DDSDDE(4,6)= EG*BBAR(5)/TWO
     1 + EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q3 
     2 + (4*EX*HGO_k1*q1*q2**2*q3)/JJ**(SE) 
     3 + (8*EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     4 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*q2**2*q3)/JJ**(FI)
		
        DDSDDE(5,6)= EG*BBAR(4)/TWO
     1 + EX*(IA/JJ**(SE) - JJ**(-(FI)))*HGO_k1*q1*q2 
     2 + (4*EX*HGO_k1*q1*q2*q3**2)/JJ**(SE) 
     3 + (8*EX*(IA/JJ**(FO) - JJ**(-(TW)))*(-1 
     4 + IA/JJ**(TW))*HGO_k1*HGO_k2*q1*q2*q3**2)/JJ**(FI)
	 		
      END IF
      DO K_1=1, NTENS
        DO K_2=1, K_1-1
          DDSDDE(K_1, K_2)=DDSDDE(K_2, K_1)
        END DO
      END DO
C
      RETURN
      END