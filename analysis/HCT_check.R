library(dplyr)
library(tidyverse)

# HCT Check
test_hct <- function(df) {
  triads <- list(
    list(cols = c("A_B", "B_C", "A_C"), name = "ABC"),
    list(cols = c("A_B", "B_D", "A_D"), name = "ABD"),
    list(cols = c("A_B", "B_E", "A_E"), name = "ABE"),
    list(cols = c("A_C", "C_D", "A_D"), name = "ACD"),
    list(cols = c("A_C", "C_E", "A_E"), name = "ACE"),
    list(cols = c("A_D", "D_E", "A_E"), name = "ADE"),
    list(cols = c("B_C", "C_D", "B_D"), name = "BCD"),
    list(cols = c("B_C", "C_E", "B_E"), name = "BCE"),
    list(cols = c("B_D", "D_E", "B_E"), name = "BDE"),
    list(cols = c("C_D", "D_E", "C_E"), name = "CDE")
  )
  
  checked_df <- df
  
  for (triad in triads) {
    columns <- triad$cols
    column_name <- triad$name
    
    # Probability vector extraction for triads
    p_ab <- checked_df[[columns[1]]]
    p_bc <- checked_df[[columns[2]]]
    p_ac <- checked_df[[columns[3]]]
    
    # checking for ANY valid HCT 
    hct_transitive <-
      (p_ab >= 0.9 & p_bc >= 0.9 & p_ac >= 0.9) |                 # A > B > C
      (p_ac >= 0.9 & (1 - p_bc) >= 0.9 & p_ab >= 0.9) |           # A > C > B  
      ((1 - p_ab) >= 0.9 & p_ac >= 0.9 & p_bc >= 0.9) |           # B > A > C 
      (p_bc >= 0.9 & (1 - p_ac) >= 0.9 & (1 - p_ab) >= 0.9) |     # B > C > A
      ((1 - p_ac) >= 0.9 & p_ab >= 0.9 & (1 - p_bc) >= 0.9) |     # C > A > B 
      ((1 - p_bc) >= 0.9 & (1 - p_ab) >= 0.9 & (1 - p_ac) >= 0.9) # C > B > A 
    
    
    # Aggregate Check 
    hct_condition <- hct_transitive 
    
    # Result added to specific triad to corresponding triad 
    checked_df[[column_name]] <- as.integer(hct_condition)
  }
  
  # Getting name of all new 'hct_...' columns 
  hct_cols <- sapply(triads, function(t) t$name)
  # If every HCT adherence, insert 1 otherwise 0 
  checked_df$HCT_check_total <- ifelse(rowSums(checked_df[, hct_cols]) == 10, 1, 0)
  
  return(checked_df)
}