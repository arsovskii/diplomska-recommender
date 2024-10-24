<script lang="ts">
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { recommendationsStore } from '$lib/stores/stores';
	import RatingStars from './ratingStars.svelte';
	import Loader from './loader.svelte';
	import Book from './book.svelte';
	import anime from 'animejs';

	interface Book {
		id: number;
		title: string;
		author: string;
		category: string;
		countReviews: number;
		rating: number;
		image: string;
	}

	export let books: Book[] = [];
	let hasMounted = false;
	let carouselContainer: HTMLElement;
	let scrollContainer: HTMLElement;
	let isDragging = false;
	let startX: number;
	let scrollLeft: number;
	let isUpdating = false;

	let user_id = 1;

	const DRAG_SENSITIVITY = 1.5;
	const SCROLL_SPEED_MULTIPLIER = 3;

	recommendationsStore.subscribe(async (newRecommendations) => {
		console.log(newRecommendations)
		if (newRecommendations && newRecommendations.length > 0) {
			if (hasMounted) {
				// Animate out current books
				await animateOutBooks();
				books = newRecommendations;
				// Animate in new books
				await animateInBooks();
			} else {
				books = newRecommendations;
			}
		}
	});

	const animateOutBooks = async () => {
		isUpdating = true;
		await anime({
			targets: '.book-card',
			scale: [1, 0.8],
			opacity: [1, 0],
			translateY: [0, 50],
			duration: 200,
			easing: 'easeInOutQuad',
			delay: anime.stagger(30)
		}).finished;
	};

	const animateInBooks = async () => {
		await anime({
			targets: '.book-card',
			scale: [0.8, 1],
			opacity: [0, 1],
			translateY: [-50, 0],
			duration: 200,
			easing: 'easeOutQuad',
			delay: anime.stagger(30)
		}).finished;
		isUpdating = false;
	};

	const getRecommendations = async () => {
		try {
			const response = await fetch('http://localhost:5000/api/recommend', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ user_id })
			});
			const text = await response.text();
			if (response.ok) {
				const data = JSON.parse(text);
				books = data.books;
			} else {
				console.error('Failed to fetch recommendations:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching recommendations:', error);
		}
	};

	// Mouse and touch event handlers remain the same as previous version
	function handleMouseDown(e: MouseEvent) {
		isDragging = true;
		startX = e.pageX - scrollContainer.offsetLeft;
		scrollLeft = scrollContainer.scrollLeft;
		scrollContainer.style.cursor = 'grabbing';
		scrollContainer.style.userSelect = 'none';
		scrollContainer.classList.add('dragging');
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isDragging) return;
		e.preventDefault();
		const x = e.pageX - scrollContainer.offsetLeft;
		const walk = (x - startX) * DRAG_SENSITIVITY * SCROLL_SPEED_MULTIPLIER;
		scrollContainer.scrollLeft = scrollLeft - walk;
	}

	function handleMouseUp() {
		if (!isDragging) return;
		isDragging = false;
		scrollContainer.style.cursor = 'grab';
		scrollContainer.style.userSelect = '';
		scrollContainer.classList.remove('dragging');
		snapToNearestBook();
	}

	function handleTouchStart(e: TouchEvent) {
		isDragging = true;
		startX = e.touches[0].pageX - scrollContainer.offsetLeft;
		scrollLeft = scrollContainer.scrollLeft;
		scrollContainer.classList.add('dragging');
	}

	function handleTouchMove(e: TouchEvent) {
		if (!isDragging) return;
		const x = e.touches[0].pageX - scrollContainer.offsetLeft;
		const walk = (x - startX) * DRAG_SENSITIVITY * SCROLL_SPEED_MULTIPLIER;
		scrollContainer.scrollLeft = scrollLeft - walk;
	}

	function handleTouchEnd() {
		if (!isDragging) return;
		isDragging = false;
		scrollContainer.classList.remove('dragging');
		snapToNearestBook();
	}

	function snapToNearestBook() {
		const bookWidth = 256; // Updated to match new card width (w-64)
		const gap = 24;
		const itemWidth = bookWidth + gap;
		const scrollPosition = scrollContainer.scrollLeft;
		const nearestItem = Math.round(scrollPosition / itemWidth);

		scrollContainer.scrollTo({
			left: nearestItem * itemWidth,
			behavior: 'smooth'
		});
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const scrollAmount = e.deltaY * SCROLL_SPEED_MULTIPLIER;
		scrollContainer.scrollLeft += scrollAmount;

		clearTimeout(scrollContainer.dataset.scrollTimeout as any);
		scrollContainer.dataset.scrollTimeout = setTimeout(snapToNearestBook, 150).toString();
	}

	const slidyMount = async () => {
		if (!hasMounted) return;
		carouselContainer.style.display = 'block';
		await anime({
			targets: '.book-card',
			opacity: [0, 1],
			scale: [0.9, 1],
			translateY: [20, 0],
			duration: 200,
			easing: 'easeOutQuad',
			delay: anime.stagger(20)
		}).finished;
	};

	onMount(() => {
		(async () => {
			await getRecommendations();
			await slidyMount();

			scrollContainer.addEventListener('wheel', handleWheel, { passive: false });
		})();
		hasMounted = true;

		return () => {
			scrollContainer?.removeEventListener('wheel', handleWheel);
		};
	});
</script>

<div
	class="relative w-full max-w-[90vw] mx-auto px-4 py-8"
	bind:this={carouselContainer}
	style="display: none;"
>
	<div
		bind:this={scrollContainer}
		class="scroll-container flex gap-6 overflow-x-auto pb-4 cursor-grab scroll-smooth"
		on:mousedown={handleMouseDown}
		on:mousemove={handleMouseMove}
		on:mouseup={handleMouseUp}
		on:mouseleave={handleMouseUp}
		on:touchstart={handleTouchStart}
		on:touchmove={handleTouchMove}
		on:touchend={handleTouchEnd}
	>
		{#each books as book (book.id)}
			<Book {book}></Book>
		{/each}
	</div>
</div>

{#if !hasMounted}
	<div class="w-full h-full relative">
		<Loader />
	</div>
{/if}

<style>
	.scroll-container {
		scroll-snap-type: x mandatory;
		scrollbar-width: thin;
	}

	.scroll-container::-webkit-scrollbar {
		height: 6px;
	}

	.scroll-container::-webkit-scrollbar-track {
		@apply bg-gray-100 rounded-full;
	}

	.scroll-container::-webkit-scrollbar-thumb {
		@apply bg-gray-300 rounded-full hover:bg-gray-400 transition-colors;
	}

	.scroll-container.dragging {
		scroll-behavior: auto;
		cursor: grabbing !important;
	}

	.scroll-container.dragging .book-card {
		pointer-events: none;
	}

	.book-card {
		scroll-snap-align: start;
	}

	/* Glass effect for rating container */
	.rating-container {
		@apply backdrop-blur-sm bg-white/80 rounded-lg shadow-sm;
	}
</style>
