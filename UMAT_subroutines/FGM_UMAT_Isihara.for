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
      REAL*8 C_10, C_01, C_20, D1_inv, L, R0, R, SIG
      PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0, FOUR=4.D0)
      PARAMETER(NINE=9.D0, EIGHT = 8.D0)


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
      DO K1=1, 3
        DO K2=1, 3
          DISTGR(K2, K1)=SCALE*DFGRD1(K2, K1)
        END DO
      END DO
	  
      DO K1=1, 3
        DO K2=1, 3
          DISTGR_VAL(K2, K1)= DFGRD1(K2, K1)
        END DO
      END DO	  
	  

      C10_MAT = 2.0D0
      C10_CORE = 6.0D0

	  PI = 4.0D0*ATAN(1.0D0)

      IF (KINC .EQ. 1 .AND. KSTEP .EQ. 1) THEN

c		Smooth Bimaterial (Diffused Sphere Inclusion)
		  R0 = 12.0D0
		  SIG = 4.0D0
		  R = SQRT((X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2)
		  STATEV(1)=C10_MAT+(C10_CORE-C10_MAT)*0.50D0*(1.0D0-TANH((R-R0)/SIG))
		  STATEV(2) = 0.5D0*STATEV(1)
C		  Mooney-Rivlin >> C_20 = STATEV(3) = 0	  
		  STATEV(3) = 0.0D0*STATEV(1)
		  STATEV(4) = 10.0D0*STATEV(1)
		  
      END IF

	  C_10 = STATEV(1)
      C_01 = STATEV(2)
	  C_20 = STATEV(3)
      D1_inv  = STATEV(4)	  

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
      EG=TWO*C_10/DET
      EK=TWO*D1_inv*(TWO*DET-ONE)
      PR=TWO*D1_inv*(DET-ONE)
	  PW = DET**(-2.D0/3.D0)
	  PWW = DET**(-5.D0/3.D0)
	  PWWW = DET**(-7.D0/3.D0)
	  PZ = DET**(2.D0/3.D0)
	  PFT = DET**(-4.D0/3.D0)
	  
	  C_FN = FOUR/NINE
	  C_FT = FOUR/THREE
	  C_ET = EIGHT/THREE
	  T_2 = C_01
	  T_3 = C_20
	  
      DO K1=1,NDI
        STRESS(K1)= EG*(BBAR(K1)-TRBBAR)+PR 
     1 + 2.D0*C_01*PWWW*(-BVAL_SQ(K1) + BVAL(K1)*TRBVAL
     2 - (1.D0/3.D0)*(TRBVAL**2 - TRBVAL_SQ))
     3 + 4.D0*C_20*PWW*( BVAL(K1) - TRBVAL/3.D0)*(TRBVAL*PW - 3.D0)
      END DO
      DO K1=NDI+1,NDI+NSHR
        STRESS(K1)=EG*BBAR(K1) 
     1 + 2.D0*C_01*PWWW*(-BVAL_SQ(K1) + BVAL(K1)*TRBVAL)	 
     2 + 4.D0*C_20*PWW*( BVAL(K1) )*(TRBVAL*PW - 3.D0)
      END DO
C
C    CALCULATE THE STIFFNESS
C
      EG23=EG*TWO/THREE
      DDSDDE(1, 1)= EG23*(BBAR(1)+TRBBAR)+EK 
     1 - T_2*PWWW*(4.D0/9.D0)*(BVAL(4)**2 + BVAL(5)**2 
     2 - BVAL(1)*BVAL(2) + 4.D0*BVAL(6)**2 - BVAL(1)*BVAL(3)
     3 - 4.D0*BVAL(2)*BVAL(3))	 
     4 +T_3*(8.D0/9.D0)*PWWW*(8.D0*BVAL(1)**2+BVAL(1)*(BVAL(2)+ BVAL(3))
     5 + 2.D0*(BVAL(2) + BVAL(3))**2 
     6 - 3.D0*PZ*(4.D0*BVAL(1)+ BVAL(2) + BVAL(3)))	 
	  
      DDSDDE(2, 2)= EG23*(BBAR(2)+TRBBAR)+EK
     1 -T_2*PWWW*(4.D0/9.D0)*(BVAL(4)**2 + 4.D0*BVAL(5)**2 
     2 - BVAL(1)*BVAL(2) + BVAL(6)**2 - 4.D0*BVAL(1)*BVAL(3)
     3 - BVAL(2)*BVAL(3))
     4 +T_3*(8.D0/9.D0)*PWWW*( 2.D0*BVAL(1)**2 + 8.D0*BVAL(2)**2
     5 + BVAL(2)*BVAL(3) + 2.D0*BVAL(3)**2 
     6 - 3.D0*PZ*(BVAL(1) + 4.D0*BVAL(2) + BVAL(3)) + BVAL(1)*(BVAL(2)
     7 + 4.D0*BVAL(3)))	 
	  
      DDSDDE(3, 3)= EG23*(BBAR(3)+TRBBAR)+EK
     1 -T_2*PWWW*(4.D0/9.D0)*(4.D0*BVAL(4)**2 + 1.D0*BVAL(5)**2 
     2 - 4.D0*BVAL(1)*BVAL(2) + BVAL(6)**2 - 1.D0*BVAL(1)*BVAL(3)
     3 - BVAL(2)*BVAL(3))
     4 +T_3*(8.D0/9.D0)*PWWW*(2.D0*BVAL(1)**2 + 2.D0*BVAL(2)**2 
     5 + BVAL(2)*BVAL(3) + 8.D0*BVAL(3)**2 
     6 + BVAL(1)*(4.D0*BVAL(2) + BVAL(3))
     7 - 3.D0*PZ*(BVAL(1) + BVAL(2) + 4.D0*BVAL(3)))
	 
      DDSDDE(1, 2)=-EG23*(BBAR(1)+BBAR(2)-TRBBAR)+EK
     1 -T_2*PWWW*(4.D0/9.D0)*(BVAL(4)**2 - 2.D0*BVAL(5)**2 
     2 - 1.D0*BVAL(1)*BVAL(2) - 2.D0*BVAL(6)**2 + 2.D0*BVAL(1)*BVAL(3)
     3 + 2.D0*BVAL(2)*BVAL(3))
     4 +T_3*(8.D0/9.D0)*PWWW*( -4.D0*BVAL(1)**2 -4.D0*BVAL(2)**2
     5 + PZ*(6.D0*BVAL(1) + 6.D0*BVAL(2) - 3.D0*BVAL(3))
     6 + BVAL(1)*(BVAL(2) - 2.D0*BVAL(3))
     7 - 2.D0*BVAL(2)*BVAL(3) + 2.D0*BVAL(3)**2)	 
	  	  
      DDSDDE(1, 3)=-EG23*(BBAR(1)+BBAR(3)-TRBBAR)+EK
     1 +T_2*PWWW*(4.D0/9.D0)*(2.D0*BVAL(4)**2 - 1.D0*BVAL(5)**2 
     2 - 2.D0*BVAL(1)*BVAL(2) + 2.D0*BVAL(6)**2 + 1.D0*BVAL(1)*BVAL(3)
     3 - 2.D0*BVAL(2)*BVAL(3))
     4 + T_3*(8.D0/9.D0)*PWWW*(-4.D0*BVAL(1)**2 
     5 + BVAL(1)*(-2.D0*BVAL(2) + BVAL(3)) 
     6 + PZ*(6.D0*BVAL(1) - 3.D0*BVAL(2) + 6.D0*BVAL(3))
     7 + 2.D0*(BVAL(2)**2 - BVAL(2)*BVAL(3) - 2.D0*BVAL(3)**2))	 
	  	  
      DDSDDE(2, 3)=-EG23*(BBAR(2)+BBAR(3)-TRBBAR)+EK
     1 +T_2*PWWW*(4.D0/9.D0)*(2.D0*BVAL(4)**2 + 2.D0*BVAL(5)**2 
     2 - 2.D0*BVAL(1)*BVAL(2) - 2.D0*BVAL(6)**2 - 2.D0*BVAL(1)*BVAL(3)
     3 + 1.D0*BVAL(2)*BVAL(3))
     4 -T_3*(8.D0/9.D0)*PWWW*(-2.D0*BVAL(1)**2 + 4.D0*BVAL(2)**2
     5 - BVAL(2)*BVAL(3) + 4.D0*BVAL(3)**2
     6 + 2.D0*BVAL(1)*(BVAL(2) + BVAL(3)) 
     7 + 3.D0*PZ*(BVAL(1) - 2.D0*(BVAL(2) + BVAL(3))))	 
	  	  
      DDSDDE(1, 4)= EG23*BBAR(4)/TWO
     1 +T_2*PWWW*(2.D0/3.D0)*(1.D0*BVAL(5)*BVAL(6)-1.D0*BVAL(4)*BVAL(3))
     2 - T_3*(4.D0/3.D0)*PWWW*BVAL(4)*(3.D0*PZ - 5.D0*BVAL(1) 
     3 + BVAL(2) + BVAL(3))

      DDSDDE(2, 4)= EG23*BBAR(4)/TWO
     1 +T_2*PWWW*(2.D0/3.D0)*(1.D0*BVAL(5)*BVAL(6)-1.D0*BVAL(4)*BVAL(3))	  
     2 - T_3*(4.D0/3.D0)*PWWW*BVAL(4)*(3.D0*PZ + BVAL(1) 
     3 - 5.D0*BVAL(2) + BVAL(3))
	  
      DDSDDE(3, 4)=-EG23*BBAR(4)
     1 -T_2*PWWW*(4.D0/3.D0)*(1.D0*BVAL(5)*BVAL(6)-1.D0*BVAL(4)*BVAL(3))
     2 +T_3*(8.D0/3.D0)*PWWW*BVAL(4)*(3.D0*PZ - 2.D0*BVAL(1)
     3 -2.D0*BVAL(2)+BVAL(3))
	 	  
      DDSDDE(4, 4)= EG*(BBAR(1)+BBAR(2))/TWO
     1 + T_2*PWWW*(-BVAL(5)**2 - BVAL(6)**2
     2 + BVAL(3)*(BVAL(1) + BVAL(2)))	
     3 + T_3*2.D0*PWWW*(BVAL(1)**2 + 4.D0*BVAL(4)**2 + BVAL(2)**2
     4 - 3.D0*PZ*(BVAL(1) + BVAL(2)) + BVAL(2)*BVAL(3)
     5 + BVAL(1)*(2.D0*BVAL(2) + BVAL(3)))
	 
	  
      IF(NSHR.EQ.3) THEN
        DDSDDE(1, 5)= EG23*BBAR(5)/TWO
     1 -T_2*PWWW*(2.D0/3.D0)*(BVAL(5)*BVAL(2)-BVAL(4)*BVAL(6))
     2 -T_3*(4.D0/3.D0)*PWWW*BVAL(5)*(3.D0*PZ
     3 -5.D0*BVAL(1)+BVAL(2)+BVAL(3))	 
		
        DDSDDE(2, 5)=-EG23*BBAR(5)
     1 +T_2*PWWW*(4.D0/3.D0)*(BVAL(5)*BVAL(2)-BVAL(4)*BVAL(6))		
     2 +T_3*(8.D0/3.D0)*PWWW*BVAL(5)*(3.D0*PZ-2.D0*BVAL(1)+BVAL(2)
     3 -2.D0*BVAL(3))	
	
        DDSDDE(3, 5)= EG23*BBAR(5)/TWO
     1 -T_2*PWWW*(2.D0/3.D0)*(BVAL(5)*BVAL(2)-BVAL(4)*BVAL(6))	
     2 - T_3*(4.D0/3.D0)*PWWW*BVAL(5)*(3.D0*PZ 
     3 + BVAL(1)+BVAL(2)-5.D0*BVAL(3))		
		
        DDSDDE(1, 6)=-EG23*BBAR(6)
     1 -T_2*PWWW*(4.D0/3.D0)*(BVAL(4)*BVAL(5)-BVAL(1)*BVAL(6))	
     2 +T_3*(8.D0/3.D0)*PWWW*BVAL(6)*(3.D0*PZ+BVAL(1)
     3 -2.D0*(BVAL(2)+BVAL(3)))		
		
        DDSDDE(2, 6)= EG23*BBAR(6)/TWO
     1 +T_2*PWWW*(2.D0/3.D0)*(BVAL(4)*BVAL(5)-BVAL(1)*BVAL(6))		
     2 - T_3*(4.D0/3.D0)*PWWW*BVAL(6)*(3.D0*PZ 
     3 +BVAL(1)-5.D0*BVAL(2)+BVAL(3))	
	
        DDSDDE(3, 6)= EG23*BBAR(6)/TWO
     1 +T_2*PWWW*(2.D0/3.D0)*(BVAL(4)*BVAL(5)-BVAL(1)*BVAL(6))		
     2 -T_3*(4.D0/3.D0)*PWWW*BVAL(6)*(3.D0*PZ 
     3 +BVAL(1)+ BVAL(2)-5.D0*BVAL(3))		
	
        DDSDDE(5, 5)= EG*(BBAR(1)+BBAR(3))/TWO
     1 + T_2*PWWW*(-BVAL(4)**2 + BVAL(1)*BVAL(2) -BVAL(6)**2 
     2 + BVAL(2)*BVAL(3))		
     3 + T_3*2.D0*PWWW*(BVAL(1)**2 + 4.D0*BVAL(5)**2 + BVAL(2)*BVAL(3)
     4 + BVAL(3)**2 - 3.D0*PZ*(BVAL(1) +BVAL(3))
     5 + BVAL(1)*(BVAL(2) + 2.D0*BVAL(3)))	 
			
        DDSDDE(6, 6)= EG*(BBAR(2)+BBAR(3))/TWO
     1 + T_2*PWWW*(-BVAL(4)**2 - BVAL(5)**2 
     2 + BVAL(1)*(BVAL(2) + BVAL(3)))		
     3 + T_3*2.D0*PWWW*(BVAL(2)**2+4.D0*BVAL(6)**2+2.D0*BVAL(2)*BVAL(3)
     4 + BVAL(3)**2 - 3.D0*PZ*(BVAL(2) + BVAL(3)) 
     5 + BVAL(1)*(BVAL(2) + BVAL(3)))
		
        DDSDDE(4,5)= EG*BBAR(6)/TWO
     1 + T_2*PWWW*(BVAL(4)*BVAL(5) - BVAL(1)*BVAL(6))
     2 + T_3*PWWW*(8.D0*BVAL(4)*BVAL(5)
     3 + 2.D0*BVAL(6)*(-3.D0*PZ + BVAL(1) + BVAL(2) + BVAL(3)))	 
		
        DDSDDE(4,6)= EG*BBAR(5)/TWO
     1 + T_2*PWWW*(-BVAL(5)*BVAL(2) + BVAL(4)*BVAL(6))	
     2 + T_3*2.D0*PWWW*(-3.D0*PZ*BVAL(5)+BVAL(1)*BVAL(5)+BVAL(5)*BVAL(2)
     3 + 4.D0*BVAL(4)*BVAL(6) + BVAL(5)*BVAL(3)) 
		
        DDSDDE(5,6)= EG*BBAR(4)/TWO
     1 + T_2*PWWW*(BVAL(5)*BVAL(6) - BVAL(4)*BVAL(3))	
     2 + T_3*2.D0*PWWW*(-3.D0*PZ*BVAL(4)+BVAL(1)*BVAL(4)+BVAL(4)*BVAL(2)
     3 + 4.D0*BVAL(5)*BVAL(6) + BVAL(4)*BVAL(3)) 
	 		
      END IF
      DO K1=1, NTENS
        DO K2=1, K1-1
          DDSDDE(K1, K2)=DDSDDE(K2, K1)
        END DO
      END DO
C
      RETURN
      END