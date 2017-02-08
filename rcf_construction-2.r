library(dplyr)

#### fichier observations
REFobs <- "H:/_C2-Restricted-Access/Database/All/v2016-07-18"
fobs   <- file.path(REFobs, "fac-obs.rdata")

load(fobs)

#### filtrages 

# obs exploitables
obs <- obs %>% 
       filter(!is.na(commit.CUR) & commit.CUR > 0)

# modele revolver sur 5 obs ou plus pour se qualifier
# age première obs <= 0.02
obs <- mutate(obs,
              duree = as.numeric(maturity-effective),
              age = as.numeric(photo-effective)/duree)

rcfn <- group_by(obs, uid) %>%
        summarize(nb = sum( model == "Revolvers stat" ),
                  first.age = min(age))

xtabs(~ (nb >= 5) + (first.age <= 0.02), rcfn)
#          first.age <= 0.02
# nb >= 5 FALSE  TRUE
#   FALSE 20019 17394
#   TRUE   1202  4734

summary(rcfn)

uid <- rcfn$uid[ (rcfn$nb >= 5) & (rcfn$first.age <= 0.02) ] # 4734 facilités

fac <- fac[fac$uid %in% uid,]
obs <- obs[obs$uid %in% fac$uid,]

summary(obs)

##### ajout infos
fidx <- match(obs$uid, fac$uid)

obs <- mutate(obs,
              tirage  = pmin(on/pmax(100,commit),1),
              on      = tirage * commit,
              on.CUR  = tirage * commit.CUR,
              off     = pmax(off, 0),
              off.CUR = pmax(off.CUR, 0),
              
              last.obs = fac$last.obs[fidx],
              first.obs = fac$first.obs[fidx],
              last.active = fac$last.active[fidx],
              arret = fac$arrêt[fidx],
              type.arret = fac$type.arrêt[fidx],
              age.droite = as.numeric(last.active-effective)/duree,
              age.arret = as.numeric(arret-effective)/duree,
              age.gauche = as.numeric(first.obs-effective)/duree )

obs <- mutate(obs,
              age.gauche = pmax(age.gauche,0),
              age.arret = ifelse(age.arret < 0.02, NA, age.arret) )


### mise au propre

obs <- mutate(obs,
              age    = pmax(0., pmin(1., age)),
              tirage = pmax(0., pmin(1., tirage)) )

summary(obs)

# lst <- fac$uid[ fac$first.obs < fac$effective + 20 ]
# obs <- obs[obs$uid %in% lst,]
# obs1 <- filter(obs, uid %in% lst) %>%
#         transmute(uid, 
#                   age = pmax(0., pmin(1., age)),
#                   tirage = pmax(0., pmin(1., tirage))) %>%
#         arrange(uid, age)


#### interpolation

sx = 0:49 / 49
lst <- unique(obs$uid)


# très très long
system.time( {
obs2 <- list()
for(u in lst[1:100]) {
  tmp <- filter(obs, uid==u)
  if(nrow(tmp) > 1) {
    obs2[[u]] <- data.frame(uid = u,
                       x = 0:49 / 49 ,
                       y = approx(tmp$age, tmp$tirage, 0:49 / 49, rule=2)$y)
  }  
  cat("+")
}  } )
#  user  system elapsed 
# 36.56    8.75   45.49 

tobs <- obs[ obs$uid %in% lst[1:100],]

proc <- function(age, tirage, arret) {
  v = approx(age, tirage, 0:49 / 49, rule=2:1 )$y
  if( ! is.na(arret) ) {
    v[is.na(v)] <- 0.
  }
  v  
}

system.time({
res <- group_by(tobs, uid) %>%
       do( res = proc(.$age, .$tirage, .$arret[1]) )
})
# user  system elapsed 
# 8.91    1.81   10.74 

# i = 3
# filter(fac, uid==res[[1]][[i]])
# res[[2]][[i]]
# res2[[2]][[i]]

# as.list(res[[2]])
# as.data.frame(res[[2]], optional=TRUE)
# tst <- as.matrix(res[[2]], optional=TRUE)


system.time({
  res <- arrange(obs, uid, age) %>%
         group_by(uid) %>%
         do( res = proc(.$age, .$tirage, .$arret[1]) )
})
# 45 minutes

save(res, file="c:/temp/rcf2.rdata")


resmat <- as.data.frame(res[[2]])

write.table(resmat, file="c:/temp/rcf_matrix2.csv", 
            col.names=FALSE, row.names=FALSE)


