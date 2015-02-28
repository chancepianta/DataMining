library(shiny)

shinyUI(fluidPage(
		titlePanel("Katrina.csv"),

		sidebarLayout(
				sidebarPanel(
						sliderInput(
								"bins",
								"Number of bins:",
								min = 6,
								max = 18,
								value = 12
							),
            br(),
            radioButtons(
                "dist", 
                "Variables to explore:",
                c(
                    "Flood Depth" = "flood_depth",
                    "Log Median Income" = "log_medinc"
                  )
            )
					),
				mainPanel(
					plotOutput("distPlot")
				)
			)
		)
	)