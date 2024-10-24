<script lang="ts">
	import { ratingsStore, recommendationsStore } from '$lib/stores/stores';
	import { fade, fly, scale } from 'svelte/transition';
	import { elasticOut } from 'svelte/easing';
	import anime from 'animejs';
	import { CheckCircle, AlertCircle, RefreshCw } from 'lucide-svelte';

	let ratingsToSend: any;
	let isLoading = false;
	let progressValue = 0;
	let progressAnimation: anime.AnimeInstance;
	let error: string | null = null;
	let showSuccess = false;
	let startTime: number;

	ratingsStore.subscribe((value) => {
		ratingsToSend = value;
	});

	const startLoadingAnimation = () => {
		progressValue = 0;
		error = null;
		startTime = Date.now();

		progressAnimation = anime({
			targets: '.progress-bar',
			width: ['0%', '90%'], // Only go to 90% initially
			duration: 10000, // Longer duration as we'll speed it up
			easing: 'easeInOutQuad',
			loop: true,
			update: (anim: { progress: number; }) => {
				progressValue = anim.progress * 0.9;
			}
		});
	};

	const stopLoadingAnimation = () => {
		const elapsedTime = Date.now() - startTime;
		const minimumAnimationTime = 800; // Minimum time to show animation

		if (progressAnimation) {
			progressAnimation.pause();
		}

		// Calculate remaining duration to ensure smooth completion
		const completionDuration = Math.max(minimumAnimationTime - elapsedTime, 400);

		// Animate to 100% smoothly
		anime({
			
			duration: completionDuration,
			easing: 'easeOutQuad',
			update: (anim) => {
				progressValue = 90 + anim.progress * 0.1; // Animate from 90% to 100%
			},
			complete: () => {
				setTimeout(() => {
					isLoading = false;
					progressValue = 0;
					showSuccessAnimation();
				}, 200);
			}
		});
	};

	const showSuccessAnimation = () => {
		showSuccess = true;
		anime({
			targets: '.success-icon',
			scale: [0, 1],
			rotate: ['20deg', '0deg'],
			duration: 800,
			easing: 'spring(1, 80, 10, 0)'
		});
		setTimeout(() => {
			showSuccess = false;
		}, 2000);
	};

	const handleError = (message: string) => {
		error = message;
		anime({
			targets: '.error-container',
			translateY: [20, 0],
			opacity: [0, 1],
			duration: 400,
			easing: 'easeOutCubic'
		});
	};

	const getRecommendations = async () => {
		isLoading = true;
		startLoadingAnimation();

		try {
			const response = await fetch('http://localhost:5000/api/makeRecommendation', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ ratings: ratingsToSend })
			});
			const text = await response.text();

			if (response.ok) {
				const data = JSON.parse(text);
				recommendationsStore.set(data.preds);
				stopLoadingAnimation();
			} else {
				throw new Error('Failed to fetch recommendations');
			}
		} catch (error) {
			console.error('Error fetching recommendations:', error);
			stopLoadingAnimation();
			handleError("Couldn't generate recommendations. Please try again.");
		}
	};

	const completeLoadingAnimation = () => {
		const elapsedTime = Date.now() - startTime;
		const minimumAnimationTime = 800; // Minimum time to show animation

		if (progressAnimation) {
			progressAnimation.pause();
		}

		// Calculate remaining duration to ensure smooth completion
		const completionDuration = Math.max(minimumAnimationTime - elapsedTime, 400);

		// Animate to 100% smoothly
		anime({
			targets: '.progress-bar',
			width: '100%',
			duration: completionDuration,
			easing: 'easeOutQuad',
			update: (anim) => {
				progressValue = Math.round(90 + anim.progress * 0.1); // Animate from 90% to 100%
			},
			complete: () => {
				setTimeout(() => {
					isLoading = false;
					progressValue = 0;
					showSuccessAnimation();
				}, 200);
			}
		});
	};

	const retryRecommendations = () => {
		error = null;
		getRecommendations();
	};

	$: loadingMessage =
		progressValue < 33
			? 'Analyzing your ratings...'
			: progressValue < 66
				? 'Finding similar readers...'
				: progressValue < 90
					? 'Curating your personal book list...'
					: 'Finalizing your recommendations...';
</script>

<div class="w-full text-center">
	{#if !isLoading && !showSuccess && !error}
		<button
			class="btn btn-secondary shadow-lg mx-auto text-xl font-normal transition-all hover:scale-105 disabled:opacity-50"
			on:click={getRecommendations}
			disabled={Object.keys(ratingsToSend || {}).length === 0}
		>
			Get your recommendations!
		</button>
	{/if}

	{#if isLoading}
		<div class="space-y-4" in:fly={{ y: 20, duration: 400 }}>
			<div class="flex flex-col items-center gap-2">
				<div class="text-lg font-medium">Generating your personalized recommendations...</div>
				<div class="text-sm text-gray-500">{Math.round(progressValue)}% complete</div>
			</div>

			<!-- Progress bar container -->

			<progress class="progress progress-secondary w-64 h-2" value={progressValue} max="100"
			></progress>

			<!-- Loading message -->
			<div class="text-sm text-gray-600 h-6" transition:fade>
				{loadingMessage}
			</div>

			
		</div>
	{/if}

	{#if showSuccess}
		<div
			class="flex items-center justify-center gap-2"
			in:scale={{ duration: 400, easing: elasticOut }}
		>
			<CheckCircle class="success-icon w-6 h-6 text-green-500" />
			<span class="text-lg text-green-600">Recommendations generated successfully!</span>
		</div>
	{/if}

	{#if error}
		<div class="error-container space-y-3" transition:fade>
			<div class="flex items-center justify-center gap-2 text-red-500">
				<AlertCircle class="w-6 h-6" />
				<span class="text-lg">{error}</span>
			</div>
			<button class="btn btn-outline btn-error btn-sm gap-2" on:click={retryRecommendations}>
				<RefreshCw class="w-4 h-4" />
				Try Again
			</button>
		</div>
	{/if}

	{#if !isLoading && Object.keys(ratingsToSend || {}).length === 0}
		<div class="text-sm text-gray-500 mt-2" transition:fade>
			Rate some books first to get personalized recommendations
		</div>
	{/if}
</div>

<style>
	

	/* Add smooth transitions for all animations */
	button {
		transition: all 0.2s ease-in-out;
	}

	/* Disable button hover effects while loading */
	button:disabled {
		cursor: not-allowed;
		transform: none;
	}

	/* Success icon animation */
	

	/* Error container animation */
	.error-container {
		transform-origin: top center;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
</style>
