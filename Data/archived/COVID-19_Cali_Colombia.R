# Spatio-temporal locations of 38617 positive cases of COVID-19 occurred in the 
# city of Cali - Colombia between March 15 and September 30 of 2020, the region 
# of interest is formed by the polygon of the political and administrative 
# boundaries of this city which contains all locations.

# Source: 
# "Secretaría de Salud Pública, Alcaldía de Santiago de Cali, Cali, Colombia"
# https://www.cali.gov.co/salud/"

## Memory cleaning
rm(list=ls())

## Set the working space
# setwd("")

##Load the RData file
#load("")

## Load the R Packages
library(rgdal)
library(rgl)
library(maptools)
library(spatstat)
library(plot3D)

## Read the csv file with the report of the positive cases of COVID-19 in the Cali city
c19cc <- read.csv(file = "COVID-19_Cali_Colombia.csv")
class(c19cc)
dim(c19cc)
head(c19cc)

## Read the shapefile to create the Cali city contour
cali <- as.owin(readShapeSpatial("Cali-Colombia.shp"))
window <- owin(poly=list(x=cali$bdry[[1]]$x,y=cali$bdry[[1]]$y))
dataF <- as.matrix(as.data.frame(window))
coordB <- project(dataF,"+proj=utm +zone=18N ellps=WGS84")

## Rescaling of the observation window
cali.cont <- owin(poly=list(x = coordB[,1]/1000,y=coordB[,2]/1000))

## Times
day <- as.POSIXct(c19cc$fec_not, format = "%d/%m/%Y", tz = "America/Chicago")
# Set a starting point
Starts <- ISOdate(year = 2020, month = 03, day = 15, hour = 0,
                  tz = "America/Chicago")
# Establish the time unit
TimeUnit <- "days" # "weeks"
# Do the dirty work to gain a nice continuous time interval
time <- as.numeric(difftime(day, Starts, units = TimeUnit))

## Create the orderly spatio-temporal point patterns

# Project and jitter coordinates to avoid duplicated point
dataC <- as.matrix(data.frame(x=c19cc$x,y=c19cc$y))

# Transform to UTM coordinates first
coordA <- project(dataC,"+proj=utm +zone=18N ellps=WGS84")+
  rnorm(length(c19cc$x),0,0.5)

# Create the orderly spatio-temporal point pattern
all.pp <- ppp(x=coordA[,1]/1000,y=coordA[,2]/1000,window=cali.cont,marks=time)
all.pp # original pattern
ok <- inside.owin(x=all.pp$x,y=all.pp$y,w=cali.cont)
in.stpp <- data.frame(x=all.pp$x[ok],y=all.pp$y[ok],marks=all.pp$marks[ok])
dup.spp <- duplicated(in.stpp)
pp.c19 <- ppp(x=in.stpp$x[!dup.spp],y=in.stpp$y[!dup.spp],window=cali.cont,
              marks=in.stpp$marks[!dup.spp])
pp.c19

# Spatio-temporal point pattern
covid <- data.frame(x=pp.c19$x,y=pp.c19$y,t=pp.c19$marks)

# Data
head(covid)

# Number of points
length(covid[,3])

# Temporal interval
range(covid[,3])

# Spatial area
area(cali.cont)
source(url("http://www.math.mcmaster.ca/bolker/R/misc/scatter3d.R"))
## Spatio-temporal point pattern plot
par(mfrow=c(1,1))
scatter3d(covid[,1],covid[,2],covid[,3],zlab="\n days",
          main="(x,y,t)-locations of the positive cases of COVID-19 in Cali, Colombia",
          theta=45,phi=30,cex=0.2,ticktype="detailed")

## Plot of the spatial point pattern
plot(unmark(pp.c19), main="Spatial locations of the positive cases of COVID-19 in Cali, 
     Colombia")

# Plot of the accumulative cases per day
plot(sort(pp.c19$marks),seq(1,length(pp.c19$marks)),type="l",xlab="t = days",
     ylab="",main="Cumulative number of the positive cases of COVID-19 in Cali, Colombia",
     las=1,xlim=range(pp.c19$marks))