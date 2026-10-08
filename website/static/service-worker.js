/* ============================================================
   PRECISION AGRICULTURE SMART ROVER
   SERVICE WORKER
   PWA + OFFLINE APP SHELL + LIVE AI DATA
============================================================ */


/* ============================================================
   VERSION
============================================================ */

const CACHE_NAME = "smart-rover-v5";


/* ============================================================
   APP SHELL
============================================================ */

const APP_SHELL = [

    "/",

    "/static/manifest.json",

    "/static/service-worker.js"

];


/* ============================================================
   INSTALL
============================================================ */

self.addEventListener(
    "install",
    function(event) {

        console.log(
            "Smart Rover Service Worker: INSTALL"
        );


        event.waitUntil(

            caches
                .open(CACHE_NAME)
                .then(function(cache) {

                    console.log(
                        "Caching Smart Rover app shell..."
                    );


                    return cache.addAll(
                        APP_SHELL
                    );

                })
                .catch(function(error) {

                    console.error(
                        "App shell caching failed:",
                        error
                    );

                })

        );


        /*
         * Activate immediately.
         */

        self.skipWaiting();

    }
);


/* ============================================================
   ACTIVATE
============================================================ */

self.addEventListener(
    "activate",
    function(event) {

        console.log(
            "Smart Rover Service Worker: ACTIVATE"
        );


        event.waitUntil(

            caches
                .keys()
                .then(function(cacheNames) {

                    return Promise.all(

                        cacheNames.map(
                            function(cacheName) {

                                if (
                                    cacheName !==
                                    CACHE_NAME
                                ) {

                                    console.log(
                                        "Deleting old cache:",
                                        cacheName
                                    );


                                    return caches.delete(
                                        cacheName
                                    );

                                }

                            }
                        )

                    );

                })
                .then(function() {

                    return self.clients.claim();

                })

        );

    }
);


/* ============================================================
   FETCH HANDLER
============================================================ */

self.addEventListener(
    "fetch",
    function(event) {

        const request =
            event.request;


        /*
         * Only process GET requests.
         */

        if (
            request.method !== "GET"
        ) {

            return;

        }


        const url =
            new URL(
                request.url
            );


        /*
         * Only handle our own website.
         */

        if (
            url.origin !==
            self.location.origin
        ) {

            return;

        }


        /* =====================================================
           1. NATIVE PASS-THROUGH FOR VIDEO STREAM & LIVE APIS
           
           NEVER intercept multipart/x-mixed-replace or live APIs!
           Service workers do not support infinite MJPEG streams.
           Passing through lets the browser stream at 25+ FPS natively.
        ===================================================== */

        if (
            url.pathname === "/video_feed" ||
            url.pathname.startsWith("/video_feed") ||
            url.pathname.startsWith("/api/")
        ) {

            return;

        }


        /* =====================================================
           2. CAMERA / AI IMAGES
           
           Never serve an old camera image when the server
           is available.
        ===================================================== */

        if (
            url.pathname.startsWith(
                "/static/images/"
            )
        ) {

            event.respondWith(

                fetch(
                    request,
                    {
                        cache: "no-store"
                    }
                )
                .catch(function() {

                    /*
                     * If offline, try the cached image
                     * if one happens to exist.
                     */

                    return caches.match(
                        request
                    );

                })

            );


            return;

        }


        /* =====================================================
           3. SERVICE WORKER
        ===================================================== */

        if (
            url.pathname ===
            "/service-worker.js"
        ) {

            event.respondWith(

                fetch(
                    request,
                    {
                        cache: "no-store"
                    }
                )

            );


            return;

        }


        /* =====================================================
           4. NORMAL WEBSITE / APP FILES
           
           NETWORK FIRST:
           
           Server available:
               → newest website
               → save a copy
           
           Server unavailable:
               → use cached website
        ===================================================== */

        event.respondWith(

            fetch(
                request
            )
            .then(function(response) {

                /*
                 * Only cache successful responses.
                 */

                if (
                    response &&
                    response.status === 200 &&
                    response.type === "basic"
                ) {

                    const responseClone =
                        response.clone();


                    caches
                        .open(
                            CACHE_NAME
                        )
                        .then(function(cache) {

                            cache.put(
                                request,
                                responseClone
                            );

                        })
                        .catch(function(error) {

                            console.log(
                                "Cache update error:",
                                error
                            );

                        });

                }


                return response;

            })
            .catch(function() {

                console.log(
                    "Server unavailable. Loading cached app:"
                    ,
                    request.url
                );


                /*
                 * Try exact cached page first.
                 */

                return caches
                    .match(request)
                    .then(function(
                        cachedResponse
                    ) {

                        if (
                            cachedResponse
                        ) {

                            return cachedResponse;

                        }


                        /*
                         * If exact page isn't cached,
                         * load the cached dashboard.
                         */

                        return caches.match(
                            "/"
                        )
                        .then(function(
                            dashboardResponse
                        ) {

                            if (
                                dashboardResponse
                            ) {

                                return dashboardResponse;

                            }


                            /*
                             * Nothing available.
                             */

                            return new Response(

                                `
                                <!DOCTYPE html>

                                <html>

                                <head>

                                <meta charset="UTF-8">

                                <meta
                                    name="viewport"
                                    content="width=device-width,
                                             initial-scale=1.0"
                                >

                                <title>
                                    Smart Rover Offline
                                </title>

                                <style>

                                body {

                                    margin: 0;

                                    min-height: 100vh;

                                    display: flex;

                                    align-items: center;

                                    justify-content: center;

                                    font-family: Arial,
                                                 sans-serif;

                                    background: #f4f7f5;

                                    color: #12372a;

                                    text-align: center;

                                }

                                .box {

                                    background: white;

                                    padding: 35px;

                                    border-radius: 18px;

                                    max-width: 400px;

                                    margin: 20px;

                                    box-shadow:
                                        0 10px 30px
                                        rgba(
                                            20,
                                            60,
                                            40,
                                            0.10
                                        );

                                }

                                .icon {

                                    font-size: 55px;

                                    margin-bottom: 15px;

                                }

                                h1 {

                                    font-size: 22px;

                                    margin-bottom: 10px;

                                }

                                p {

                                    color: #6b7280;

                                    line-height: 1.6;

                                    font-size: 14px;

                                }

                                </style>

                                </head>

                                <body>

                                <div class="box">

                                    <div class="icon">
                                        🌱
                                    </div>

                                    <h1>
                                        Smart Rover Offline
                                    </h1>

                                    <p>
                                        The Smart Rover server
                                        is currently unavailable.
                                    </p>

                                    <p>
                                        Start the Flask server
                                        and reconnect to access
                                        live AI and rover data.
                                    </p>

                                </div>

                                </body>

                                </html>
                                `,

                                {

                                    status: 200,

                                    headers: {

                                        "Content-Type":
                                            "text/html"

                                    }

                                }

                            );

                        });

                    });

            })

        );

    }
);


/* ============================================================
   MESSAGE HANDLER
============================================================ */

self.addEventListener(
    "message",
    function(event) {

        console.log(
            "Message received by service worker:",
            event.data
        );


        /*
         * Allow the page to request
         * immediate activation.
         */

        if (
            event.data &&
            event.data.type ===
            "SKIP_WAITING"
        ) {

            self.skipWaiting();

        }

    }
);


/* ============================================================
   SERVICE WORKER ERROR LOGGING
============================================================ */

self.addEventListener(
    "error",
    function(event) {

        console.error(
            "Smart Rover Service Worker error:",
            event.error
        );

    }
);


/* ============================================================
   FINAL MESSAGE
============================================================ */

console.log(
    "=========================================="
);

console.log(
    "PRECISION AGRICULTURE SMART ROVER"
);

console.log(
    "SERVICE WORKER v3 LOADED"
);

console.log(
    "LIVE API = NO CACHE"
);

console.log(
    "CAMERA IMAGES = LIVE"
);

console.log(
    "APP SHELL = CACHED"
);

console.log(
    "OFFLINE MODE = ENABLED"
);

console.log(
    "=========================================="
);