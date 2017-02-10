library(dplyr)

#### fichier observations
REFobs <- "H:/_C2-Restricted-Access/Database/All/v2016-07-18"
fobs   <- file.path(REFobs, "fac-obs.rdata")

load(fobs)

summary(obs$model)

obs <- filter(obs, model=="Revolvers stat")

summary(obs)

obs <- obs %>% filter(!is.na(commit.CUR) & commit.CUR > 0)

uid <- unique(obs$uid[obs$model == "Revolvers stat"])
fac <- fac[fac$uid %in% uid,]
obs <- obs[obs$uid %in% fac$uid,]

obs$tirage <- pmin(obs$on/pmax(100,obs$commit),1)
#obs$on <- obs$tirage * obs$commit
#obs$on.CUR <- obs$tirage * obs$commit.CUR
#obs$off <- pmax(obs$off, 0)
#obs$off.CUR <- pmax(obs$off.CUR, 0)

obs$last.obs <- fac$last.obs[match(obs$uid,fac$uid)]
obs$first.obs <- fac$first.obs[match(obs$uid,fac$uid)]
obs$last.active <- fac$last.active[match(obs$uid,fac$uid)]
obs$arrêt <- fac$arrêt[match(obs$uid,fac$uid)]
obs$type.arret <- fac$type.arrêt[match(obs$uid,fac$uid)]

obs$duree <- as.numeric(obs$maturity-obs$effective)
obs$age <- as.numeric(obs$photo-obs$effective)/obs$duree
obs$age.droite <- as.numeric(obs$last.active-obs$effective)/obs$duree
obs$age.arret <- as.numeric(obs$arrêt-obs$effective)/obs$duree
obs$age.gauche <- as.numeric(obs$first.obs-obs$effective)/obs$duree
obs$age.gauche[obs$age.gauche < 0] <- 0

obs$age.arret[obs$age.arret < 0.02] <- NA

# on en voit le début
mean(fac$first.obs < fac$effective + 20) # 80% des RCF, 4800 RCF
lst <- fac$uid[ fac$first.obs < fac$effective + 20 ]
obs <- obs[obs$uid %in% lst,]
obs1 <- filter(obs, uid %in% lst) %>%
        transmute(uid, 
                  age = pmax(0., pmin(1., age)),
                  tirage = pmax(0., pmin(1., tirage))) %>%
        arrange(uid, age)

obs2 <- list()
for(u in lst) {
  tmp <- filter(obs, uid==u)
  if(nrow(tmp) > 1) {
    obs2[[u]] <- data.frame(uid = u,
                       x = 0:49 / 49 ,
                       y = approx(tmp$age, tmp$tirage, 0:49 / 49, rule=2)$y)
  }  
  cat("+")
}
save(obs2, file="c:/temp/obs2.rdata")
obs3 <- rbind_all(obs2)

write.csv(obs3, file="c:/temp/rcf_matrix.csv")


