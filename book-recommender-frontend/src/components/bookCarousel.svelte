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
	let lastScrollTime = 0;
	let isUpdating = false;
	let currentScrollAnimation: anime.AnimeInstance | null = null;
	let velocity = 0;
	let lastX = 0;
	let rafId: number | null = null;
	let favoriteGenre = '';

	let user_id = 1;

	// Константи за брзина на скролирање
	const MOMENTUM_FACTOR = 0.95;
	const VELOCITY_THRESHOLD = 0.5;

	// Ги следиме промените во recommendationsStore, ако се случи ажурирање на книгите ги изнесуваме со анимација старите, а новите ги внесуваме
	recommendationsStore.subscribe(async (newRecommendations) => {
		console.log(newRecommendations);
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
	// Анимација за излез на книгите
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

	// Анимација за влез на книгите
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

	// Функција за добивање на првични препораки
	const getRecommendations = async () => {
		console.log(favoriteGenre);
		try {
			const response = await fetch('http://localhost:5000/api/recommend', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ "user_id":user_id, "genres": favoriteGenre })
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

	// функција за справување со клик на листата со препораки
	function handleMouseDown(e: MouseEvent) {
		isDragging = true;
		startX = e.pageX;
		lastX = e.pageX;
		scrollLeft = scrollContainer.scrollLeft;
		velocity = 0;

		if (currentScrollAnimation) {
			currentScrollAnimation.pause();
		}

		if (rafId) {
			cancelAnimationFrame(rafId);
			rafId = null;
		}

		scrollContainer.style.cursor = 'grabbing';
		scrollContainer.style.userSelect = 'none';
		scrollContainer.classList.add('dragging');
	}

	// функција за справување со движење на препорачаните книги
	function handleMouseMove(e: MouseEvent) {
		if (!isDragging) return;
		e.preventDefault();

		const x = e.pageX;
		const dx = x - lastX;
		velocity = dx;
		lastX = x;

		const walk = startX - x;
		scrollContainer.scrollLeft = scrollLeft + walk;
	}

	// функција за примена на забрзувањето на скролирањето
	function applyMomentum() {
		if (Math.abs(velocity) < VELOCITY_THRESHOLD) {
			velocity = 0;
			rafId = null;
			snapToNearestBook();
			return;
		}

		scrollContainer.scrollLeft -= velocity;
		velocity *= MOMENTUM_FACTOR;
		rafId = requestAnimationFrame(applyMomentum);
	}

	// функција за справување со отпуштање
	function handleMouseUp() {
		if (!isDragging) return;
		isDragging = false;
		scrollContainer.style.cursor = 'grab';
		scrollContainer.style.userSelect = '';
		scrollContainer.classList.remove('dragging');

		if (Math.abs(velocity) > VELOCITY_THRESHOLD) {
			rafId = requestAnimationFrame(applyMomentum);
		} else {
			snapToNearestBook();
		}
	}


	// 	функции за справување со допир на екран
	function handleTouchStart(e: TouchEvent) {
		isDragging = true;
		startX = e.touches[0].pageX;
		lastX = e.touches[0].pageX;
		scrollLeft = scrollContainer.scrollLeft;
		velocity = 0;

		if (currentScrollAnimation) {
			currentScrollAnimation.pause();
		}

		if (rafId) {
			cancelAnimationFrame(rafId);
			rafId = null;
		}

		scrollContainer.classList.add('dragging');
	}

	function handleTouchMove(e: TouchEvent) {
		if (!isDragging) return;
		const x = e.touches[0].pageX;
		const dx = x - lastX;
		velocity = dx;
		lastX = x;

		const walk = startX - x;
		scrollContainer.scrollLeft = scrollLeft + walk;
	}

	function handleTouchEnd() {
		if (!isDragging) return;
		isDragging = false;
		scrollContainer.classList.remove('dragging');

		if (Math.abs(velocity) > VELOCITY_THRESHOLD) {
			rafId = requestAnimationFrame(applyMomentum);
		} else {
			snapToNearestBook();
		}
	}

	// функција за приближување на најблиската книга
	function snapToNearestBook() {
		const bookWidth = 256;
		const gap = 24;
		const itemWidth = bookWidth + gap;
		const scrollPosition = scrollContainer.scrollLeft;
		const nearestItem = Math.round(scrollPosition / itemWidth);

		if (currentScrollAnimation) {
			currentScrollAnimation.pause();
		}

		currentScrollAnimation = anime({
			targets: scrollContainer,
			scrollLeft: nearestItem * itemWidth,
			duration: 600,
			easing: 'cubicBezier(0.4, 0.0, 0.2, 1)'
		});
	}

	// променлива за чување на долготрајната акумулирана вредност, за полесно скролирање
	let accumulatedDelta = 0;
	const WHEEL_SENSITIVITY = 0.5;

	function handleWheel(e: WheelEvent) {
		e.preventDefault();

		// Accumulate the delta for smoother scrolling
		accumulatedDelta += e.deltaY * WHEEL_SENSITIVITY;

		// Clear any existing animation
		if (currentScrollAnimation) {
			currentScrollAnimation.pause();
		}

		if (rafId) {
			cancelAnimationFrame(rafId);
		}

		// Use RAF for smooth scrolling
		rafId = requestAnimationFrame(() => {
			scrollContainer.scrollLeft += accumulatedDelta;
			accumulatedDelta = 0;

			// Set up snap timeout
			clearTimeout(scrollContainer.dataset.scrollTimeout as any);
			scrollContainer.dataset.scrollTimeout = setTimeout(snapToNearestBook, 150).toString();
		});
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

	// При mount на компонентата, добиваме препораки и ги прикажуваме
	onMount(() => {
		(async () => {
			const storedGenre = localStorage.getItem('favoriteGenres');
			if (storedGenre) {
				favoriteGenre = storedGenre;
			}

			await getRecommendations();
			await slidyMount();
			if(scrollContainer){
				scrollContainer.addEventListener('wheel', handleWheel, { passive: false });	
			}

		})();
		hasMounted = true;

		return () => {
			if (rafId) cancelAnimationFrame(rafId);
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
		class="scroll-container flex gap-6 overflow-x-auto pb-4 cursor-grab will-change-scroll"
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
		scrollbar-width: thin;
		-webkit-overflow-scrolling: touch;
		scroll-behavior: auto !important;
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
		scroll-behavior: auto !important;
		cursor: grabbing !important;
	}

	.scroll-container.dragging .book-card {
		pointer-events: none;
	}

	.book-card {
		transform: translateZ(0);
	}

	/* Glass effect for rating container */
	.rating-container {
		@apply backdrop-blur-sm bg-white/80 rounded-lg shadow-sm;
	}
</style>
